package com.stack;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.examples.svm;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.tools.Statistics;

public class TestSvmSigKnn {

	@Test
	public void test() throws IOException, InvalidInputDataException {
//		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
//		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
//		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
		String trainfile = "test.txt";
		String testfile = "test.txt";
		
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		int[][] trainStat = Statistics.getLabelsStatistic(train.y);
		
		Parameter param = new Parameter(1, 3000, 0.001);
//		int[] k = {3, 5, 7, 9, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		int[] k = {3};
		double[] c = new double[15];
		double[] gamma = new double[15];
		for(int i = 0; i < 15; i++) {
			c[i] = Math.pow(2, i - 7);
			gamma[i] = c[i];
		}
		
		SvmSigKnn svmknn = new SvmSigKnn(train, param);
		double perf = Double.POSITIVE_INFINITY;
		double bestc = Double.POSITIVE_INFINITY;
		double[][] ks = null;
		for(int i = 0; i < c.length; i++) {
		    for(int j = 0; j < gamma.length; j++) {
		    	double tc = c[i];
		    	double tg = gamma[j];
		    	svmknn.crossValidation(train, param, 5, tc, tg, k);
		    }
		}
		
		param.setC(bestc);
		DataPoint[][] w = svmknn.train(train, param);
		
		int[][] pre = svmknn.getPredictLabels(train, test, w);
		double microf1 = Measures.microf1(svmknn.getUniqueLabels(), test.y, pre);
		double macrof1 = Measures.macrof1(svmknn.getUniqueLabels(), test.y, pre);
		double hammingloss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + 
				", Macro-F1 = " + macrof1 + ", Hamming Loss = " + hammingloss + 
				", 0/1 loss = " + zeroneloss);
		
		System.out.println();
		for(int i = 0; i < ks.length; i++) {
			for(int j = 0; j < ks[i].length; j++) {
				System.out.print(ks[i][j] + " ");
			}
			System.out.println();
		}
		
		for(int i = 0; i < trainStat.length; i++) {
			for(int j = 0; j < trainStat[i].length; j++) {
				System.out.print(trainStat[i][j] + " ");
			}
			System.out.println();
		}
		
		System.out.println();
		int[] ul = svmknn.getUniqueLabels();
		for(int i = 0; i < ul.length; i++) {
			System.out.print(ul[i] + " " );
		}
	}

}
