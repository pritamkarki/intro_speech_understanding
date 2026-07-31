import unittest
import os, sys

scriptdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scriptdir)
import homework11

wavefile = os.path.join(scriptdir, 'speech_waveform.wav')

# TestSequence
class Test(unittest.TestCase):       
    def test_method_returns_str(self):
        text = homework11.transcribe_wavefile(wavefile, 'en')
        self.assertIs(type(text), str, 'transcribe_wavefile should return a str, not a %s'%(type(text)))

    def test_method_works_with_known_input(self):
        text = homework11.transcribe_wavefile(wavefile, 'en')
        self.assertIn('speech',text,'transcribe_wavefile output should include the word "speech"')
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))
                                
                    
                        

