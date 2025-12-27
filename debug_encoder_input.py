import torch
import sys
sys.path.insert(0, './uer')

# Ver si vm es usado en encoder
from uer.encoders.bert_encoder import BertEncoder

# Verificar firma del forward()
import inspect
sig = inspect.signature(BertEncoder.forward)
print("BertEncoder.forward signature:")
print(sig)
print("\nParameters:")
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")
