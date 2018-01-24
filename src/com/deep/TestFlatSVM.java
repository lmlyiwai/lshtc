package com.deep;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.rssvm.Measures;

public class TestFlatSVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
//		String trainfile = "F:\\java\\RecursiveRegularizationSVM_1\\scene_train.txt";
//		String testfile = "F:\\java\\RecursiveRegularizationSVM_1\\scene_test.txt";
		
		
		String trainfile = "F:\\java\\RecursiveRegularizationSVM_1\\yeast_train.txt";
		String testfile = "F:\\java\\RecursiveRegularizationSVM_1\\yeast_test.txt";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		Parameter param = new Parameter(1, 1000, 0.001);
		double[] c = new double[15];
		for(int i = 0; i < c.length; i++) {
			c[i] = Math.pow(2, i-7);
		}
		
		FlatSVM fs = new FlatSVM(train, param);
		for(int i = 0; i < c.length; i++) {
			param.setC(c[i]);
			fs.crossValidation(train, param, 5);
		}
		
		param.setC(1);
		fs.train(train, param);
		
		int[][] pl = fs.predict(test.x);
		double microf1 = Measures.microf1(fs.getUlabels(), test.y, pl);
		double macrof1 = Measures.macrof1(fs.getUlabels(), test.y, pl);
		double hamminloss = Measures.averageSymLoss(test.y, pl);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " 
				+ macrof1 + ", Hamming Loss = " + hamminloss);
	}

}
