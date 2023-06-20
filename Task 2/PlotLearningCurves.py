import numpy as np
import matplotlib.pyplot as plt

class LearningCurvePlotter():
    dataPath = "C:\\Users\\tatam\\Desktop\\results"
    plotTitle = "Learning to Play SupermarioBros2-v0"
    dataRuns = []

    def __init__(self): 
        for run_id in range(1,3+1):
            results = self.readDataFile(run_id)
            results = self.normaliseData(results)
            self.dataRuns.append(results)
        self.plotPerformance()

    def readDataFile(self, run_id):
        fileName = self.dataPath+"\\result-ddqn-supermariobros2-run"+str(run_id)+".txt"
        print("Trying to open file="+str(fileName))
        print("HERE..."+str(fileName))
        data = []
        num_lines = 0
        try:
            with open(fileName) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line.find("INFO:")==0 and line.find(":outdir:")>0:
                        tokens = line.split(' ')
                        _tuple = []
                        for i in range(1,len(tokens)):
                            pair = tokens[i].split(':')
                            if len(pair)==1 or len(pair[1])==0: continue
                            _tuple.append(float(pair[1]))
                        data.append(_tuple)
                        num_lines += 1
        except:
            print("WARNING: File "+str(fileName)+" does not exist!")
        
        return data
        
    def normaliseData(self, unnormalisedData):
        normalisedData = []
        batchRewards = []
        for i in range(0,len(unnormalisedData)):
            step = unnormalisedData[i][0]
            episode = unnormalisedData[i][1]
            batchRewards.append(unnormalisedData[i][2])
            if i>0 and (i%20==0 or i==len(unnormalisedData)-1):
                avgReward = np.mean(batchRewards)
                _tuple = [step, episode, avgReward]
                normalisedData.append(_tuple)
                batchRewards = []
                print("i="+str(i)+" _tuple="+str(_tuple))
                
        return normalisedData

    def plotPerformance(self):
        self.data_run1 = np.array(self.dataRuns[0])
        self.data_run2 = np.array(self.dataRuns[1])
        self.data_run3 = np.array(self.dataRuns[2])
        if len(self.data_run1)>0:
            print("WOKRING PLOT")
            print("|self.data_run1[:,0]|="+str(len(self.data_run1[:,0])))
        if len(self.data_run2)>0:
            print("|self.data_run2[:,1]|="+str(len(self.data_run2[:,1])))
        if len(self.data_run3)>0:
            print("|self.data_run3[:,2]|="+str(len(self.data_run3[:,2])))
        
        fig, axs = plt.subplots(2)
        fig.suptitle(self.plotTitle)
        axs[0].set(xlabel='Step', ylabel='Avg. Reward')
        axs[1].set(xlabel='Episode', ylabel='Avg. Reward')
        
        if len(self.data_run1)>0:
            axs[0].plot(self.data_run1[:,0], self.data_run1[:,2], 'tab:red', label='Run1', linestyle='-')
            axs[1].plot(self.data_run1[:,1], self.data_run1[:,2], 'tab:red', label='Run1', linestyle='-')

        if len(self.data_run2)>0:
            axs[0].plot(self.data_run2[:,0], self.data_run2[:,2], 'tab:green', label='Run2', linestyle='--')
            axs[1].plot(self.data_run2[:,1], self.data_run2[:,2], 'tab:green', label='Run2', linestyle='--')

        if len(self.data_run3)>0:
            axs[0].plot(self.data_run3[:,0], self.data_run3[:,2], 'tab:blue', label='Run3', linestyle=':')
            axs[1].plot(self.data_run3[:,1], self.data_run3[:,2], 'tab:blue', label='Run3', linestyle=':')

        axs[0].legend()
        axs[1].legend()
        plt.show()
        
LearningCurvePlotter()
