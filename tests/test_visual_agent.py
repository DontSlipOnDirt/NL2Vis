
try:
    from core.const import REVIEWER_NAME, MAX_REVIEW_ROUNDS
    print("Constants loaded.")
    
    from core.config import VISION_VLLM_MODEL_NAME
    print(f"Config loaded: {VISION_VLLM_MODEL_NAME}")
    
    from core.agents import Reviewer, Validator, Processor, Composer
    print("Agents imported.")
    
    r = Reviewer()
    print("Reviewer instantiated.")
    
    import matplotlib.pyplot as plt
    v = Validator("dummy_path")
    
    # Test plotting capture
    code = "import matplotlib.pyplot as plt; plt.figure(); plt.plot([1,2], [3,4])"
    res = v._execute_python_code(code)
    if res.get('image_base64'):
        print("Image capture successful.")
    else:
        print("Image capture FAILED.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
