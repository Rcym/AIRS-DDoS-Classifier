import pandas as pd
import numpy as np
import random
import copy
from time import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

class AIRS:
    """
    :param train_set: training dataset
    :param CLONAL_RATE: the number of ARB clones that will be generated from MC best match
    :param HYPER_CLONAL_RATE: factor to multiply the clonal rate with
    :param MC_INIT_RATE: the percentage of dataset that will become memory cell
    :param MUTATION_RATE: the chance of mutation of each feature
    :param TOTAL_RESSOURCES: the ressource limit
    :param MAX_ITER: max iteration for competing for limited ressources (more iteration -> more ARB mutation)
    :param AFFINITY_THRESHOLD_SCALAR: multiply with the affinity threshold, limit to see if new MC is too close to best match
    :param KNN_K: K value for KNN classification
    """
    def __init__(self, train_set:np.ndarray, CLONAL_RATE:int, HYPER_CLONAL_RATE:int, CLASS_NUMBER:int, MC_INIT_RATE:float,
          MUTATION_RATE:float, TOTAL_RESSOURCES:int, MAX_ITER:int, AFFINITY_THRESHOLD_SCALAR:float, KNN_K:int):
        self.train_set = train_set
        self.CLONAL_RATE = CLONAL_RATE
        self.HYPER_CLONAL_RATE = HYPER_CLONAL_RATE
        self.CLASS_NUMBER = CLASS_NUMBER
        self.MC_INIT_RATE = MC_INIT_RATE
        self.MUTATION_RATE = MUTATION_RATE
        self.TOTAL_RESSOURCES = TOTAL_RESSOURCES
        self.MAX_ITER = MAX_ITER
        self.AFFINITY_THRESHOLD_SCALAR = AFFINITY_THRESHOLD_SCALAR
        self.KNN_K = KNN_K

        self.MC_POOL = {_class : [] for _class in range(self.CLASS_NUMBER)}
        self.ARB_POOL = {_class : [] for _class in range(self.CLASS_NUMBER)}
        self.MC_iterations = []

        self.MINs = np.array([0 for i in range(65)])
        self.MAXs = np.array([0 for i in range(65)])
        
        self.AFFINITY_THRESHOLD = 0.0

        self.train_set_length = len(train_set)  # for modelInfo function
        self.modelAccuracy = 0.0



    @staticmethod
    def affinity(vec1:np.ndarray, vec2:np.ndarray) -> float:
        return 1 / (1 + np.linalg.norm(vec1 - vec2))  # 1 / (1 + √Σ(a_i - b_i)²)

        # can also try : np.sqrt(np.sum((vec1 - vec2) ** 2))
        # can calculate ein 2 ways :
        # uclidistance / euclidistance + 1 : makes the distance range between 0-1 (0 close / 1 far)
        # 1 / 1 + euclidistance            : inverse euclidistance (0 low affinity (far) / 1 high affinity (close))


    # Calculating the affinity threshold (average affinity between all vectors)
    # Aproximate methode
    def affinity_threshold_sampled(self, sampleSize:int) -> float:
        aff_threshold = 0.0
        N = self.train_set.shape[0] # gets the number of rows in dataset
        for i in range(sampleSize):
            i_vec1, i_vec2 = random.sample(range(N), k=2)
            aff_threshold += self.affinity(self.train_set[i_vec1][:-1], self.train_set[i_vec2][:-1])
        return aff_threshold / sampleSize

    def affinity_threshold_sampled_average(self, sampleSize:int, repeatCount:int):
        aff_threshold_sum = 0.0
        aff_tracker = 0
        for _ in range(repeatCount):
            aff_tracker += 1
            if aff_tracker % int(repeatCount/10) == 0:
                print('progresion : {} %'.format(int(aff_tracker / repeatCount *100)))
            aff_threshold_sum += self.affinity_threshold_sampled(sampleSize)
        return aff_threshold_sum / repeatCount
    

    # deterministic methode (not used, too slow)
    def affinity_threshold_deterministic(self) -> float:
        aff_threshold = 0.0
        nb_samples = len(self.train_set)
        for i in range(nb_samples):
            for j in range(i+1, nb_samples):
                aff_threshold = self.affinity(self.train_set[i], self.train_set[j])
        return aff_threshold / ( nb_samples*(nb_samples-1)/2 )

    def calculate_affinity_threshold(self):
        print("calculating the affinity threshold...")
        self.AFFINITY_THRESHOLD = self.affinity_threshold_sampled_average(7000, 1000)
        print('Affinity threshold calculated : {}\n'.format(self.AFFINITY_THRESHOLD))




    # MC_INIT_RATE is the percentage of the dataset that will be used to initialize the MC pool
    def init_MC_pool(self):
        lenTrain = len(self.train_set)
        for _ in range(int(lenTrain * self.MC_INIT_RATE)):
            random_sample = self.train_set[np.random.randint(lenTrain)]
            self.MC_POOL[random_sample[-1]].append(MC(vector=random_sample[:-1], _class=random_sample[-1]))


    def min_res_ARB(self, _class:int):
        minRes = 1.0
        for i in range(len(self.ARB_POOL[_class])):
            # print("comparing {} <= {}".format(self.ARB_POOL[_class][i].ressources, minRes))
            if self.ARB_POOL[_class][i].ressources <= minRes:
                minRes = self.ARB_POOL[_class][i].ressources
                minResARB = self.ARB_POOL[_class][i]
                index_ARB = i
        return minResARB, index_ARB

    def get_max_stim_ARB_as_MC(self, _class:int):
        maxStim = 0.0
        for i in range(len(self.ARB_POOL[_class])):
            if self.ARB_POOL[_class][i].stimulation >= maxStim:
                maxStim = self.ARB_POOL[_class][i].stimulation
                maxStimARB = self.ARB_POOL[_class][i]
        return MC(vector=maxStimARB.vector, _class=maxStimARB._class)

    def get_max_stim_MC(self, antigene,_class:int):
        maxStim = 0.0
        for i in range(len(self.MC_POOL[_class])):
            if self.MC_POOL[_class][i].stimulate(antigene) >= maxStim:
                maxStim = self.MC_POOL[_class][i].stimulation
                maxStimMC = self.MC_POOL[_class][i]
        return maxStimMC
    
 


    def Min_Max_Stim_ARB(self, pattern:np.ndarray):
        MIN_STIM, MAX_STIM = 0.0, 1.0
        for c in self.ARB_POOL.keys():
            for ARB_cell in self.ARB_POOL.get(c):
                stim = ARB_cell.stimulate(pattern)
                if stim < MIN_STIM:
                    MIN_STIM = stim
                if stim > MAX_STIM:
                    MAX_STIM = stim
        # print('IN MIN MAX STIM FUNCTION : min={}, max={}'.format(MIN_STIM, MAX_STIM)) useless function ??
        return MIN_STIM, MAX_STIM

    def find_MIN_MAX_dataset(self):
        MINs = np.array([0 for i in range(len(self.train_set[0])-1)])
        MAXs = np.array([0 for i in range(len(self.train_set[0])-1)])

        for i in range(len(self.train_set)):
            for j in range(len(self.train_set[0][:-1])):
                if self.train_set[i][j] < MINs[j]:
                    MINs[j] = self.train_set[i][j]
                if self.train_set[i][j] > MAXs[j]:
                    MAXs[j] = self.train_set[i][j]
        return MINs, MAXs





    @staticmethod
    def updateTracker(tracker_count:int, total_count:int, interval:int, logfile):
        tracker_count += 1
        if tracker_count % int(total_count*interval/100) == 0:
            print("progression : {} / {}   -   {}%".format(tracker_count, total_count, tracker_count/total_count*100))
            logfile.flush()
        return tracker_count





    def train(self):
        startTime = time()
        print('Training Started\n')

        
        if self.AFFINITY_THRESHOLD == 0.0:
            raise Exception('Must calculate affinity threshold before training')

        MINs, MAXs = self.find_MIN_MAX_dataset()

        # MC initialisastion
        self.init_MC_pool()
        self.MC_iterations.append(copy.deepcopy(self.MC_POOL))

        logfile = open('debug.txt', 'a')


        # core training loop
        train_tracker = 0
        for antigene, _class in zip(self.train_set[:,:-1],self.train_set[:,-1]):
            # logfile.write('\n\n\n =======================  core loop : {}  =======================\n\n'.format(train_tracker))

            train_tracker = self.updateTracker(train_tracker, len(self.train_set), 5, logfile)


            # MC identification (getting the best match mc with antigene)
            if len(self.MC_POOL[_class]) == 0:
                Best_MC_match = MC(vector=antigene, _class=_class)
                self.MC_POOL[_class].append(Best_MC_match)
            else:
                Best_MC_match = self.get_max_stim_MC(antigene, _class)
                # logfile.write('BEST MC MATCH : {}\n'.format(Best_MC_match))

            

            # ARB Generation
            self.ARB_POOL[_class].append(ARB(vector=Best_MC_match.vector, _class=_class))
            Best_MC_match_STIM = Best_MC_match.stimulate(antigene)
            # logfile.write('Best_MC_match stimulation with antigene : {}'.format(Best_MC_match_STIM))

            # determining the number of max clones
            MAX_CLONES = int(self.CLONAL_RATE * self.HYPER_CLONAL_RATE * Best_MC_match_STIM)
            # logfile.write('MAX_CLONE = {} * {} * {} = {}\n'.format(self.CLONAL_RATE, self.HYPER_CLONAL_RATE, Best_MC_match_STIM, MAX_CLONES))
            # logfile.write('CLONAL EXPANSION AND COMPETING FOR RESSOURCES loop\n')
            iter = 0
            while True:
                iter += 1
                # logfile.write('\niter : {}__________________________________\n'.format(iter))
                # generating MAX_CLONES number of ARBs from the Best_MC_match
                num_clones = 0
                bestARB = max(self.ARB_POOL[_class], key=lambda cell: cell.stimulation)
                while num_clones < MAX_CLONES:
                    num_clones += 1
                    # newClone = Best_MC_match.mutate(MINs, MAXs, self.MUTATION_RATE)
                    newClone = bestARB.mutate(MINs, MAXs, self.MUTATION_RATE)
                    self.ARB_POOL[_class].append(newClone)

                

                # competition for ressources
                avgStim_in_ARB_pool= sum([x.stimulate(antigene) for x in self.ARB_POOL[_class]]) / len(self.ARB_POOL[_class])
                # logfile.write('avgStim in ARB pool = {}\n'.format(avgStim_in_ARB_pool))


                # normalizing the ressources
                MIN_STIM, MAX_STIM = self.Min_Max_Stim_ARB(antigene)
                # ressss = []
                for c in self.ARB_POOL.keys():
                    for ARB_cell in self.ARB_POOL.get(c):
                        ARB_cell.stimulation = (ARB_cell.stimulation - MIN_STIM) / (MAX_STIM - MIN_STIM)
                        ARB_cell.ressources = ARB_cell.stimulation # * self.CLONAL_RATE
                        # ressss.append(ARB_cell.ressources)
                # ressss.sort(reverse=True)
                
                # print('\nsorted array of ARB ressources : ',ressss)
                # logfile.write('sorted ARB ressources : {}\n'.format([round(float(x),3) for x in ressss]))

                res_allocated = sum([x.ressources for x in self.ARB_POOL[_class]])
                res_allowed_limit = self.TOTAL_RESSOURCES
                # logfile.write('res allocated {}\ntotal ressources allowed: {}\n'.format(res_allocated, res_allowed_limit))
    
                while res_allocated > res_allowed_limit:
                    res_to_remove = res_allocated - res_allowed_limit
                    
                    # removing the ARB with lowest ressource
                    ARB_to_remove, ARB_to_remove_Index = self.min_res_ARB(_class)
                    if ARB_to_remove.ressources <= res_to_remove:
                        self.ARB_POOL[_class].remove(ARB_to_remove)
                        res_allocated -= ARB_to_remove.ressources
                    else:
                        self.ARB_POOL[_class][ARB_to_remove_Index].ressources -= res_to_remove
                        res_allocated -= res_to_remove

                # print('checking if {} (avg_stim) > {} (aff threshold) OR if {} (iter) >= {} (max_iter)'.format(avgStim_in_ARB_pool, self.AFFINITY_THRESHOLD, iter, self.MAX_ITER))
                # logfile.write('checking if {} (avg_stim) > 0.9 OR if {} (iter) >= {} (max_iter)\n'.format(avgStim_in_ARB_pool, iter, self.MAX_ITER))
                # if (avgStim_in_ARB_pool > self.AFFINITY_THRESHOLD) or (iter >= self.MAX_ITER):
                if (avgStim_in_ARB_pool > 0.9) or (iter >= self.MAX_ITER):  # changed this to test
                    break
            
            MC_candidate = self.get_max_stim_ARB_as_MC(_class)
            MC_candidate.stimulate(antigene)

            # logfile.write('MC candidate stimulation : {}\nMC best match stimulation : {}\n'.format(MC_candidate.stimulation, Best_MC_match.stimulation))
            if MC_candidate.stimulation > Best_MC_match.stimulation:
                # if self.affinity(MC_candidate.vector, Best_MC_match.vector) < self.AFFINITY_THRESHOLD * self.AFFINITY_THRESHOLD_SCALAR:
                if self.affinity(MC_candidate.vector, Best_MC_match.vector) > 0.98:  # was < !!!!!!!!!!  chanegd this
                    self.MC_POOL[_class].remove(Best_MC_match)
                    # logfile.write('affinity difference too small -> mc best match removed\n')
                self.MC_POOL[_class].append(MC_candidate)
                # logfile.write('MC CANDIDATE ADDED\n')


            # updating the MC iteration with current pool
            # self.MC_iterations.append(copy.deepcopy(self.MC_POOL))

        # getting the final MC_POOL
        self.MC_iterations.append(copy.deepcopy(self.MC_POOL))
        endTime = time()
        self.trainingTime = endTime - startTime
        print("Training complete\n")
        logfile.close()





    # non essential functions

    @staticmethod
    def getMCInfo(mc_pool):
        extractedFeatures = []
        extractedClasses = []
        for c in mc_pool.keys():
            for cell in mc_pool[c]:
                extractedFeatures.append(cell.vector)
                extractedClasses.append(cell._class)
        return np.array(extractedFeatures), np.array(extractedClasses)

    def displayEvolution(self):
        initFeatures, initClasses = self.getMCInfo(self.MC_iterations[0])
        initFeatures_reduced_TSNE = TSNE(n_components=2).fit_transform(initFeatures)
        initFeatures_reduced_PCA = PCA(n_components=2).fit_transform(initFeatures)

        finalFeatures, finalClasses = self.getMCInfo(self.MC_iterations[-1])
        finalFeatures_reduced_TSNE = TSNE(n_components=2).fit_transform(finalFeatures)
        finalFeatures_reduced_PCA = PCA(n_components=2).fit_transform(finalFeatures)

        plt.figure(figsize=(10,10))

        # plotting the init part
        plt.subplot(2,2,1)
        plt.title('TSNE Init')
        plt.scatter(initFeatures_reduced_TSNE[:,0], initFeatures_reduced_TSNE[:,1], c=initClasses)
        plt.subplot(2,2,3)
        plt.title('PCA Init')
        plt.scatter(initFeatures_reduced_PCA[:,0], initFeatures_reduced_PCA[:,1], c=initClasses)

        # plotting the final part
        plt.subplot(2,2,2)
        plt.title('TSNE Final')
        plt.scatter(finalFeatures_reduced_TSNE[:,0], finalFeatures_reduced_TSNE[:,1], c=finalClasses)
        plt.subplot(2,2,4)
        plt.title('PCA Final')
        plt.scatter(finalFeatures_reduced_PCA[:,0], finalFeatures_reduced_PCA[:,1], c=finalClasses)

        plt.show


    def showModelInfo(self):
        trained = len(self.MC_POOL[0]) != 0

        # training time
        training_time_in_seconds, train_hour, train_minute, train_seconds = round(self.trainingTime),0,0,0
        if training_time_in_seconds > 3600:
            train_hour = training_time_in_seconds // 3600
            training_time_in_seconds = training_time_in_seconds % 3600
        if training_time_in_seconds > 60:
            train_minute = training_time_in_seconds // 60
            training_time_in_seconds = training_time_in_seconds % 60
        train_seconds = training_time_in_seconds
        timeToPrint = ''
        timeToPrint += '{} hours '.format(train_hour) if train_hour > 0 else ''
        timeToPrint += '{} minutes '.format(train_minute) if train_minute > 0 else ''
        timeToPrint += '{} seconds\n'.format(train_seconds)
        print('Model Trained in ',timeToPrint)

        print('Dataset lenght : {}   =  {} (trainset)  +  {} (testset)\n'.format(self.train_set_length + self.test_set_length, self.train_set_length, self.test_set_length))

        mcLenghts = []
        for _class in self.MC_POOL.keys():
            mcLenghts.append(len(self.MC_POOL[_class]))
        print('size of MC POOL : {}'.format(sum(mcLenghts)))
        for i in range(len(mcLenghts)):
            print('MC class {} : {}'.format(i, mcLenghts[i]))


    def getAccuracy(self) -> float:
        return self.modelAccuracy
    
    def getReport(self):
        return classification_report(self.real_values, self.pred_values, output_dict=True)
        
    def getMcPOOLasDataset(self):
        vectors = []
        classes = []
        for c in self.MC_POOL.keys():
            for cell in self.MC_POOL.get(c):
                vectors.append(cell.vector)
                classes.append(cell._class)
        return np.array(vectors), np.array(classes)










    def classify(self, antigene):
        if(len(self.MC_POOL) == 0):
            raise Exception("AIRS must be trained first")

        # stimulating all memory cells with the new antigene
        vote_array = []
        for c in self.MC_POOL.keys():
            for cell in self.MC_POOL.get(c):
                cell.stimulate(antigene)
                vote_array.append(cell)

        vote_array = np.array(list(sorted(vote_array, key=lambda cell : -cell.stimulation)))
        v = {_class : 0 for _class in range(self.CLASS_NUMBER)}

        K = min(self.KNN_K, len(vote_array))

        for x in vote_array[:K]:
            v[x._class] += 1

        maxVote = 0
        _class = 0
        for x in v.keys():
            if v[x] > maxVote:
                maxVote = v[x]
                _class = x
        return _class


    def Eval(self, test_set:np.ndarray):

        lenTest = len(test_set)
        self.test_set_length = lenTest  # for modelInfo function
        print('Evaluation Started')
        n_correct= 0
        eval_tracker = 0

        self.real_values = []
        self.pred_values = []

        for antigene, _class in zip(test_set[:,:-1], test_set[:,-1]):
            if eval_tracker % (lenTest//10) == 0:
                print('progression : {} %'.format(round((eval_tracker/lenTest*100),3)))
            eval_tracker += 1
            
            predicted_class = self.classify(antigene)
            self.real_values.append(_class)           # for confusion matrix
            self.pred_values.append(predicted_class)   # for confusion matrix
            if predicted_class == _class:
                n_correct += 1
        self.modelAccuracy = n_correct / lenTest
        print('Evaluation Finished\n')
        print('Accuracy : {} %'.format(self.modelAccuracy * 100))



    def displayConfusionMatrix(self, class_names=None, title=None, max=None, figsize=None):
        if class_names == None:
            class_names = ['DrDoS_NTP', 'TFTP', 'Benign', 'Syn', 'UDP', 'DrDoS_UDP', 'UDP-lag', 'MSSQL', 'DrDoS_MSSQL', 'DrDoS_DNS', 'DrDoS_SNMP', 'LDAP', 'DrDoS_LDAP', 'Portmap', 'NetBIOS', 'DrDoS_NetBIOS', 'UDPLag', 'WebDDoS']
        if title == None:
            title = 'Confusion Matrix'
        if max == None:
            max = 100
        if figsize == None:
            figsize = (15,10)

        matrix = np.zeros((self.CLASS_NUMBER, self.CLASS_NUMBER), dtype=int)
        for real, pred in zip(self.real_values, self.pred_values):
            matrix[int(real), pred] += 1

        fig, ax = plt.subplots(figsize=figsize)
        confMatrx = ax.matshow(matrix)
        ax.set_title(title)
        ax.set_xlabel('Predicted classes')
        ax.set_ylabel('Real classes')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)

        fig.colorbar(confMatrx)

        for (i, j), val in np.ndenumerate(matrix):
            ax.text(i, j, str(round(val*100/max, 1)), ha='center', va='center', color='black')

        plt.tight_layout( )
        plt.show()














class MC:
    def __init__(self, vector:np.ndarray, _class:int):
        self.vector = np.array(vector)
        self._class = _class
        self.stimulation = 0.0

    def __str__(self):
        return 'MC : class = {}, stimulation = {}'.format(self._class, self.stimulation)
    def __repr__(self):
        return 'MC : class = {}, stimulation = {}'.format(self._class, self.stimulation)
    
    def stimulate(self, pattern) -> float:
        self.stimulation = AIRS.affinity(self.vector, pattern)
        return self.stimulation
    
    def mutate(self, MINs, MAXs, MUTATION_RATE):
        mutated = False
        while mutated == False:
            mutated_vect = []
            for idx, feature in enumerate(self.vector):
                if random.random() <= MUTATION_RATE:
                    stddev = 0.02 * (MAXs[idx] - MINs[idx])
                    mutated_vect.append(random.gauss(feature, stddev))
                    # mutated_vect.append(random.uniform(MINs[idx], MAXs[idx]))
                    mutated = True
                else:
                    mutated_vect.append(feature)
        return ARB(vector=mutated_vect, _class=self._class)
    

class ARB:
    def __init__(self, vector, _class):
        self.vector = vector
        self._class = _class
        self.stimulation = 0.0
        self.ressources = 0

    def __str__(self):
        return 'ARB : class = {}, stimulation = {}'.format(self._class, self.stimulation)
    def __repr__(self):
        return 'ARB : class = {}, stimulation = {}'.format(self._class, self.stimulation)
    
    def stimulate(self, pattern) -> float:
        self.stimulation = AIRS.affinity(self.vector, pattern)
        return self.stimulation
    
    def mutate(self, MINs, MAXs, MUTATION_RATE):
        mutated = False
        while mutated == False:
            mutated_vect = []
            for idx, feature in enumerate(self.vector):
                if random.random() <= MUTATION_RATE:
                    stddev = 0.02 * (MAXs[idx] - MINs[idx])
                    mutated_vect.append(random.gauss(feature, stddev))
                    # mutated_vect.append(random.uniform(MINs[idx], MAXs[idx]))
                    mutated = True
                else:
                    mutated_vect.append(feature)
        return ARB(vector=mutated_vect, _class=self._class)