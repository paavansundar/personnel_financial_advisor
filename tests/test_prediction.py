"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Annotated

from model import generic_advice,share_specific_advice


def test_make_prediction():
    genericAdviceObj=generic_advice.GenericAdvice()
    answer=genericAdviceObj.chat(prompt)
    
    assert answer != None
