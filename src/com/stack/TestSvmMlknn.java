package com.stack;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.knn.MLKNN;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;

public class TestSvmMlknn {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
//		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		Parameter param = new Parameter(1, 3000, 0.001);
		int[] k = {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
		double[] c = new double[15];
		for(int i = 0; i < 15; i++) {
			c[i] = Math.pow(2, i - 7);
		}
		
		SvmMlknn svmknn = new SvmMlknn(train, param);
		for(int i = 0; i < c.length; i++) {
//		    double tc = c[i];
//			svmknn.gridSerach(train, param, 10, tc, k);
		}
		
		
		param.setC(0.0625);
		DataPoint[][] w = svmknn.train(train, param);
		svmknn.setWeight(w);
		int[][] pre = svmknn.predict(test.x, 20);
		
		double microf1 = Measures.microf1(svmknn.getUniqueLabels(), test.y, pre);
		double macrof1 = Measures.macrof1(svmknn.getUniqueLabels(), test.y, pre);
		double hammingloss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + 
				", Macro-F1 = " + macrof1 + ", Hamming Loss = " + hammingloss + 
				", 0/1 loss = " + zeroneloss);
	}

}
