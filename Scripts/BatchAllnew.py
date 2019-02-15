#python ..\ChromatinImagingV2\Scripts\BatchAllnew.py
import sys,os
#add path
workbookDir = os.getcwd()
sys.path.append(os.path.dirname(workbookDir)+os.sep+r'\CommonTools')
import IOTools as io    

if __name__ == "__main__":
    script = r'"'+workbookDir+os.sep+r'BatchSequentialSmall2colV3.py"'
    str_runs = []
    for i in range(10):
        str_runs.append('python '+script+' '+str(i))
    io.batch_command(str_runs,batch_size=10)