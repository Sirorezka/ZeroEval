from argparse import Namespace
import asyncio
from asyncio import Queue, QueueEmpty
from typing import List, Any, Callable
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from tqdm import tqdm

def create_vllm_async_engine(args: Namespace):
    max_model_len = None if args.max_model_len == -1 else args.max_model_len
    engine_args = AsyncEngineArgs(
        model=args.model_name,
        tokenizer=args.tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    # llm = None
    llm = AsyncLLMEngine.from_engine_args(engine_args)    
    return llm


async def _shutdown_engine(engine: AsyncLLMEngine | None):
    if engine is None:
        return
    
    try:
        engine.shutdown()
        print("vLLM AsyncEngine shut down gracefully.")
    except Exception as e:
        print(f"Error when closing engind: {e}")


def shutdown_vllm_async_engine(llm):
    asyncio.run(_shutdown_engine(llm))

# Define worker function
async def worker(llm: AsyncLLMEngine, 
                 sampling_params: SamplingParams, 
                 prompts: List[str], 
                 outputs: List[List[str]], 
                 queue: Queue, 
                 pbar: tqdm, 
                 worker_id: int = 0,
                 saver: Callable | None = None):
    """
    Args:
        saver (Callable | None, optional): function that dumps generated outputs to selected file.

    Raises:
        e: _description_
    """


    cnt_saved = 0
    save_interval = 10 # save every 10 generations

    while not queue.empty():
        try:
            prompt_idx = queue.get_nowait()
            prompt = prompts[prompt_idx]
        except QueueEmpty as e:
            break
        
        try:
            request_id = f"request-{prompt_idx}"

            # Get a prompt idx from the queue
            generator = llm.generate(prompt, sampling_params, request_id=request_id)

            final_result = None
            async for result in generator:
                final_result = result

            outs = [x.text for x in final_result.outputs]
            outputs[prompt_idx].extend(outs)

            pbar.update(1)
            queue.task_done()

            if worker_id == 0 and saver is not None:
                cnt_done = 0
                for x in outputs:
                    # order metters, we can't save all completed outputs
                    if len(x)>0:
                        cnt_done += 1
                    else:
                        break

                if cnt_done-cnt_saved>=save_interval:
                    saver(outputs[:cnt_done])
                    cnt_saved = cnt_done

        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")
            raise e
        

async def _run_async_inference(llm, 
                               args, 
                               sampling_params, 
                               prompts, 
                               num_workers = 10,
                               saver: Callable | None = None):
    n = len(prompts)
    num_workers = min(n, num_workers)

    workers = []
    outputs = [[] for _ in range(n)]  # prepopulate outputs
    descr = f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"
    with tqdm(total=n, desc=descr) as pbar:

        # Create an async queue and populate it with prompt indexes
        queue = asyncio.Queue(maxsize=n)
        for i in range(len(prompts)):
            queue.put_nowait(i)

        # Create and start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(worker(llm, sampling_params, prompts, outputs, queue, pbar=pbar, worker_id=i, saver=saver))
            workers.append(task)
        
        await asyncio.gather(*workers, return_exceptions=True)

    for task in workers:
        if task.exception():
            raise task.exception()
        
    return outputs


def run_vllm_async_inference(llm, 
                             args: Namespace, 
                             sampling_params: SamplingParams, 
                             prompts: List[str],
                             saver: Callable | None = None):
    """
    Run inference using the async vLLM engine.
    Prompts are processed by async workers that send them to the engine.
    """
    if len(prompts) == 0:
        return []
    
    num_workers = 4 * args.data_parallel_size
    print(f"Num async workers {num_workers}")
    outputs = asyncio.run(_run_async_inference(llm, args, sampling_params, prompts, num_workers=num_workers, saver=saver))
    return outputs

