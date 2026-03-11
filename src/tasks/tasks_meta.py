from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod
from datasets import Dataset
import datasets
import pandas as pd

# container for all existing tasks
TASKS_COLLECTION: Dict[str, TaskMeta] = {}

def register_class(cls: TaskMeta):
    # decorator for automatically registering classes
    TASKS_COLLECTION[cls.name] = cls()
    return cls


class TaskMeta(ABC):
    """A class that defines a task."""
    _id_name: str = "id" # column with uniqie_id of each dataset item
    name: str
    total_num_examples: int
    summary_file: str

    def list_all_tasks(self):
        return ",".join([x.name for x in TASKS_COLLECTION])
            
    @abstractmethod
    def load_dataset(self) -> Dataset:
        # load hf dataset
        pass

    @property
    def id_name(self):
        return self._id_name
         
    def load_summary_file(self):
        data = pd.read_json(self.summary_file)
        return data
    

@register_class
class GSMTask(TaskMeta):
     name = 'gsm'
     total_num_examples = 1319
     summary_file = 'result_dirs/gsm.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("yuchenlin/zero-eval", "gsm", split="test")
          return dataset

@register_class
class MMLUReduxTask(TaskMeta):
     name = 'mmlu-redux'
     total_num_examples = 2778
     summary_file = 'result_dirs/mmlu-redux.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
          return dataset
     

@register_class
class ZebraGridTask(TaskMeta):
     name = 'zebra-grid'
     total_num_examples = 1000
     summary_file = 'result_dirs/zebra-grid.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
          return dataset

@register_class
class AlpacaEval(TaskMeta):
     name = 'alpaca_eval'
     total_num_examples = -1
     summary_file = 'result_dirs/alpaca.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
          return dataset

@register_class
class NumerSenceV2Eval(TaskMeta):
     name = 'numersense-v2'
     total_num_examples = -1
     summary_file = 'result_dirs/numersense-v2.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("yuchenlin/zero-eval", "numersense-v2", split="test")
          return dataset

@register_class
class CruxTask(TaskMeta):
     name = 'crux'
     total_num_examples = 800
     summary_file = 'result_dirs/crux.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("flydust/zero-eval", "crux", split="test")
          return dataset

@register_class
class MathL5Task(TaskMeta):
     name = 'math-l5'
     total_num_examples = 721
     summary_file = 'result_dirs/math-l5.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("AI-MO/aimo-validation-math-level-5", split="train")
          return dataset
     
@register_class
class WildBench_V2Task(TaskMeta):
     _id_name = 'session_id'
     name = 'wildbench_v2-hard'
     total_num_examples = -1
     summary_file = 'result_dirs/wildbench_v2-hard.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("allenai/WildBench", "v2-hard", split="test")
          return dataset

@register_class
class GPlanetTask(TaskMeta):
     name = 'gplanet'
     total_num_examples = 1396
     summary_file = 'result_dirs/gplanet.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("WildEval/G-PlanET", split="test")
          return dataset


@register_class
class HendrycksMathTask(TaskMeta):
     _id_name: str = "unique_id"
     name = 'hendrycks-math'
     total_num_examples = 500
     summary_file = 'result_dirs/gplanet.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
          return dataset     