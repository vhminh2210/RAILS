try:
    from .cpp.uni_evaluator import UniEvaluator
    print("Evaluate model with cpp")
except:
    from .python.uni_evaluator import UniEvaluator
    print("Evaluate model with python")
