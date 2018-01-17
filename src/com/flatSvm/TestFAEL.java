package com.flatSvm;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.meta.MetaSvmKnn;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.tools.Statistics;

public class TestFAEL {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
//		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
//		String trainfile = "test.txt";
//		String testfile = "test.txt";
		
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		double[] c = new double[15];
		for(int i = 0; i < 15; i++) {
			c[i] = Math.pow(2, i - 7);
		}
		
		double[] g = {1, 2, 3, 4, 5, 6, 7};
		double[] lambda = {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000};
		
		FAEL fael = new FAEL(train, param);
		for(int i = 0; i < 15; i++) {
			for(int j = 0; j < lambda.length; j++) {
				for(int k = 0; k < g.length; k++) {
					fael.corssValidation(train, param, 5, c[i], g[k], lambda[j]);
				}
			}
		}	

		

	}

}
