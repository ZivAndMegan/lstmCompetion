import tensorflow as tf
import numpy
from mlxtend.preprocessing import one_hot


class DataTrainSet(object):
    def __init__(self, pathNameTrain, pathNameTest):
        self.getTrainData(pathNameTrain)
        self.getTestData(pathNameTest)

    @property
    def images_humain(self):
        return self._images_humain
    @property
    def images_robot(self):
        return self._images_robot
    @property
    def num_examples_humain(self):
        return self._num_examples_humain
    @property
    def num_examples_robot(self):
        return self._num_examples_robot

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def test_images(self):
        return self._test_images

    @property
    def test_num_examples(self):
        return self._test_num_examples

    @property
    def test_epochs_completed(self):
        return self._test_epochs_completed

    def next_batch_train(self, batch_size):
        start_humain = self._index_in_epoch_humain
        start_robot = self._index_in_epoch_robot
        self._index_in_epoch_humain += int(batch_size/2)
        self._index_in_epoch_robot += int(batch_size/2)
        if self._index_in_epoch_humain > self._num_examples_humain:
            # Shuffle the data
            perm = [i for i in range(self._num_examples_humain)]
            numpy.random.shuffle(perm)
            self._images_humain = self._images_humain[perm]
            # Start next epoch
            start_humain = 0
            self._index_in_epoch_humain = int(batch_size/2)
            assert batch_size <= self._num_examples_humain
        if self._index_in_epoch_robot > self._num_examples_robot:
            # Shuffle the data
            perm = [i for i in range(self._num_examples_robot)]
            numpy.random.shuffle(perm)
            self._images_robot = self._images_robot[perm]
            # Start next epoch
            start_robot = 0
            self._index_in_epoch_robot = int(batch_size/2)
            assert batch_size <= self._num_examples_robot
        end_humain = self._index_in_epoch_humain
        end_robot = self._index_in_epoch_robot
        train_data = numpy.concatenate((self._images_humain[start_humain:end_humain],self._images_robot[start_robot:end_robot]),axis=0)
        label_data_tem1 = [1 for i in range(end_humain-start_humain)]
        label_data_tem2 = [0 for i in range(end_robot-start_robot)]
        label_data_tem1.extend(label_data_tem2)
        label_data = one_hot(label_data_tem1)
        perm_data = [i for i in range(len(train_data))]
        numpy.random.shuffle(perm_data)
        train_data = train_data[perm_data]
        label_data = label_data[perm_data]
        return train_data,label_data

    def next_batch_test(self, batch_size):
        start = self._test_index_in_epoch
        self._test_index_in_epoch += batch_size
        if self._test_index_in_epoch > self._test_num_examples:
          # Finished epoch
            self._test_epochs_completed += 1
            # Shuffle the data
            perm = [i for i in range(self._test_num_examples)]
            numpy.random.shuffle(perm)
            self._test_images = self._test_images[perm]
            # Start next epoch
            start = 0
            self._test_index_in_epoch = batch_size
            assert batch_size <= self._test_num_examples
        end = self._test_index_in_epoch
        return numpy.array(self._test_images[start:end])

    def getTestData(self, pathName):
        f = open(pathName, 'r')
        listData = []
        line = True
        maxColone = 0
        while line:
            line = f.readline()
            myList = line.split(" ")
            if len(myList) == 3:
                myList2 = myList[1].split(";")
                myList2 = myList2[:len(myList2) - 1]
                if(maxColone < len(myList2)):
                    maxColone = len(myList2)
                myList2Tem = []
                for subMyList2 in myList2:
                    listTemp = subMyList2.split(",")
                    listTemp = list(map(float, listTemp))
                    myList2Tem.append(listTemp)
                listData.append(myList2Tem)
        f.close()
        # print (max(listMax))
        for onelist in listData:
            dataFill = onelist[-1]
            listFill = [dataFill for i in range(maxColone - len(onelist))]
            onelist.extend(listFill)
        self._test_images = numpy.array(listData)
        self._test_num_examples = len(listData)
        self._test_epochs_completed = 0
        self._test_index_in_epoch = 0

    def getTrainData(self, pathName):
        f = open(pathName, 'r')
        labels_humain = []
        labels_robot = []
        listData_humain = []
        listData_robot = []
        line = True
        maxColone = 0
        while line:
            line = f.readline()
            myList = line.split(" ")
            if len(myList) == 4:
                myList2 = myList[1].split(";")
                myList2 = myList2[:len(myList2) - 1]
                if(maxColone < len(myList2)):
                    maxColone = len(myList2)
                myList2Tem = []
                for subMyList2 in myList2:
                    listTemp = subMyList2.split(",")
                    listTemp = list(map(float, listTemp))
                    myList2Tem.append(listTemp)
                if int(myList[3][0])==1:
                    listData_humain.append(myList2Tem)
                elif int(myList[3][0])==0:
                    listData_robot.append(myList2Tem)
        f.close()
        # print (max(listMax))
        for onelist in listData_humain:
            dataFill = onelist[-1]
            listFill = [dataFill for i in range(maxColone - len(onelist))]
            onelist.extend(listFill)
        for onelist in listData_robot:
            dataFill = onelist[-1]
            listFill = [dataFill for i in range(maxColone - len(onelist))]
            onelist.extend(listFill)
        self._images_humain = numpy.array(listData_humain)
        self._images_robot = numpy.array(listData_robot)
        self._num_examples_humain = len(listData_humain)
        self._num_examples_robot = len(listData_robot)
        self._epochs_completed = 0
        self._index_in_epoch_humain = 0
        self._index_in_epoch_robot = 0
