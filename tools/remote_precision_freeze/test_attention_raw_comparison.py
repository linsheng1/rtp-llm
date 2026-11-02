import ast,types,unittest
from pathlib import Path
p=Path(__file__).resolve().parent/'analyze_capture_attention_v2.py';tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['need','compare_original_attention_point']];namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace);compare=namespace['compare_original_attention_point']
class Raw(unittest.TestCase):
 def test_FP8_original_bits(self):
  x=compare(None,bytes([0,128,255]),bytes([0,129,255]),'torch.uint8','FP8_bits');self.assertEqual((x['bitwise_unequal_elements'],x['differing_bytes']),(1,1));self.assertNotIn('max_abs_error_finite',x)
 def test_packed_scale_int32(self):
  x=compare(None,b'\x00'*8,b'\x00'*4+b'\x01\x00\x00\x80','torch.int32','scale');self.assertEqual((x['elements'],x['bitwise_unequal_elements'],x['first_bitwise_difference_column']),(2,1,1))
 def test_float_path_delegates_to_original(self):
  called=[];c=types.SimpleNamespace(numeric_report=lambda *x:called.append(x) or {'float':True});self.assertEqual(compare(c,b'\0\0',b'\0\0','torch.bfloat16','q'),{'float':True});self.assertEqual(len(called),1)
 def test_bad_alignment_rejected(self):
  with self.assertRaises(ValueError):compare(None,b'1',b'2','torch.int32','bad')
if __name__=='__main__':unittest.main()
