package com.deep;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;

public class TestYeast {

	@Test
	public void test() throws IOException, InvalidInputDataException {
//		String filename = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
//		String filename = "F:\\DataSets\\scene\\scene-train.svm";
//		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
		String filename = "F:\\DataSets\\emotion\\emotions-train.svm";
		String testfile = "F:\\DataSets\\emotion\\emotions-test.svm";
		
//		String filename = "testAnn.txt";
//		String testfile = "testAnn.txt";
		
		Problem prob = Problem.readProblem(new File(filename), 1);
		Problem testprob = Problem.readProblem(new File(testfile), 1);
		ANN ann = new ANN(prob, 4);
		ann.newTrain(prob, testprob, 0.00001, 1e-5, 50000);
//		ann.newTrain(prob, 0.0001, 1e-10, 5000);
		
		double[][] pv = new double[testprob.l][];
		int[][] pl = new int[testprob.l][];
		for(int i = 0; i < testprob.l; i++) {
			pv[i] = ann.predictFinalOutput(testprob.x[i], testprob.n);
			pl[i] = ann.predict(testprob.x[i], 0.5, testprob.n);
		}
		
		double microf1 = Measures.microf1(ann.getLabels(), testprob.y, pl);
		double macrof1 = Measures.macrof1(ann.getLabels(), testprob.y, pl);
		double hammingLoss = Measures.averageSymLoss(testprob.y, pl);
		
		System.out.println("Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + ", Hamming Loss = "
				+ hammingLoss);
	}

}
