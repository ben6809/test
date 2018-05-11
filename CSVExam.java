package test;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.*;
//import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class CSVExam {

    private static Logger log = LoggerFactory.getLogger(CSVExam.class);

    public static void main(String[] args) throws  Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
        Nd4j.create(1);

        Configuration cudaConf = CudaEnvironment.getInstance().getConfiguration();
//        cudaConf.allowMultiGPU(true);
//        cudaConf.allowCrossDeviceAccess(true);

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader trainReader = new CSVRecordReader(numLinesToSkip,delimiter);
//        trainReader.initialize(new FileSplit(new File("/home/DATA/classfication/csv/train.csv")));
        trainReader.initialize(new FileSplit(new File("/home/ubuntu/download/dl4j-1.0.0-alpha/csv/train.csv")));
        RecordReader testReader = new CSVRecordReader(numLinesToSkip,delimiter);
//        testReader.initialize(new FileSplit(new File("/home/DATA/classfication/csv/test.csv")));
        testReader.initialize(new FileSplit(new File("/home/ubuntu/download/dl4j-1.0.0-alpha/csv/test.csv")));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 2;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 10000;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainReader,batchSize,labelIndex,numClasses);
        DataSetIterator testIterator = new RecordReaderDataSetIterator(trainReader,batchSize,labelIndex,numClasses);

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        trainIterator.reset();


        final int numInputs = 2;
        int outputNum = 2;
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

//        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
//        // DataSets prefetching options. Set this value with respect to number of actual devices
//        .prefetchBuffer(32)
//        // set number of workers equal to number of available devices. x1-x2 are good values to start with
//        .workers(4)
//        // rare averaging improves performance, but might reduce model accuracy
//        .averagingFrequency(1)
//        // if set to TRUE, on every averaging model score will be reported
////        .reportScoreAfterAveraging(true)
//        .build();
        
        trainIterator.setPreProcessor(normalizer);
        for(int i=0; i<100; i++ ) {
            System.out.println("Epoch : " + (i+1));
            model.fit(trainIterator);
            trainIterator.reset();
        }

        testIterator.setPreProcessor(normalizer);
        Evaluation eval = model.evaluate(testIterator);
        
        //evaluate the model on the test set
//        Evaluation eval = new Evaluation(2);
//        INDArray output = model.output(testIterator);
//        eval.eval(testIterator.getLabels(), output);
        System.out.println(eval.stats());
        log.info(eval.stats());
    }

}

