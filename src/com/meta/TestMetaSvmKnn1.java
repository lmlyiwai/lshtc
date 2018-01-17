package com.meta;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.tools.Statistics;

public class TestMetaSvmKnn1 {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
//		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
//		String trainfile = "test.txt";
//		String testfile = "test.txt";
		
		
		Problem train = Problem.readProblem(new File(trainfile), -1);
		Problem test = Problem.readProblem(new File(testfile), -1);
		
		int[][] trainStat = Statistics.getLabelsStatistic(train.y);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		int[] k1 = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
		int[] k2 = {3, 5, 7, 9, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		double[] c = new double[15];
		for(int i = 0; i < 15; i++) {
			c[i] = Math.pow(2, i - 7);
		}
		

		
		MetaSvmKnn svmknn = new MetaSvmKnn(train, param);
			
//		svmknn.pbpu(train, param, k1, k2, c);
//		svmknn.train(train, param, 5, k1, c, k2);
		
		MetaFeature mf = new MetaFeature(train);
		double[][] trainpv = mf.transTrainSet(train, 10);
		
		double[][] testpv = mf.transTestSet(train, test, 10);
		
		train.x = mf.trans(trainpv);
		train.n = trainpv[0].length;
		
		test.x = mf.trans(testpv);
		test.n = testpv[0].length;
		
		param.setC(0.0078125);
		DataPoint[][] w = svmknn.train(train, param);
		
		double[][] trainprev = svmknn.predictValues(w, train.x, train.n);
		svmknn.scale(trainpv);
		
		double[][] testprev = svmknn.predictValues(w, test.x, test.n);
		svmknn.scale(testpv);
		
		int[][] pre = svmknn.predictNear(trainprev, testprev, train.y, 21);
		
		double microf1 = Measures.microf1(svmknn.getUniqueLabels(), test.y, pre);
		double macrof1 = Measures.macrof1(svmknn.getUniqueLabels(), test.y, pre);
		double hammingloss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + 
				", Macro-F1 = " + macrof1 + ", Hamming Loss = " + hammingloss + 
				", 0/1 loss = " + zeroneloss);
	}

}
