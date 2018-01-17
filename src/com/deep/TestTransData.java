package com.deep;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;

public class TestTransData {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\yeast\\yeast_train.svm";
		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
//		String filename = "F:\\DataSets\\scene\\scene-train.svm";
//		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
//		String filename = "F:\\DataSets\\emotion\\emotions-train.svm";
//		String testfile = "F:\\DataSets\\emotion\\emotions-test.svm";
		
//		String filename = "testAnn.txt";
//		String testfile = "testAnn.txt";
		
		Problem prob = Problem.readProblem(new File(filename), 1);
		Problem testprob = Problem.readProblem(new File(testfile), 1);
		ANN ann = new ANN(prob, 9);
		ann.newTrain(prob, testprob, 0.0001, 1e-10, 2500);
		
		double[][] trainpv = new double[prob.l][];
		for(int i = 0; i < prob.l; i++) {
			trainpv[i] = ann.predictFinalOutput(prob.x[i], prob.n);
		}
		
		double[][] testpv = new double[testprob.l][];
		for(int i = 0; i < testprob.l; i++) {
			testpv[i] = ann.predictFinalOutput(testprob.x[i], testprob.n);
		}

		String sceneTrain = "yeast_train.txt";
		String scentTes = "yeast_test.txt";
		Matrix.writeMatToFile(sceneTrain, trainpv, prob.y);
		Matrix.writeMatToFile(scentTes, testpv, testprob.y);
	}
	
	
}
