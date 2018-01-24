package com.test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.rssvm.RevisedMultiLabel;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.FileIO;
import com.tools.FileInputOutput;

public class TestRevisedMultilabel {

	@Test
	public void test() throws Exception, InvalidInputDataException {
		String trainFile = "F:\\DataSets\\rcv1\\rcv1_topics_train.svm";
		String testfile0 = "F:\\DataSets\\rcv1\\rcv1_topics_test_0.svm";
		String testfile1 = "F:\\DataSets\\rcv1\\rcv1_topics_test_1.svm";
		String testfile2 = "F:\\DataSets\\rcv1\\rcv1_topics_test_2.svm";
		String testfile3 = "F:\\DataSets\\rcv1\\rcv1_topics_test_3.svm";
		
		Problem prob = FileIO.readProblem(new File(trainFile), 1);	
		
		Map<Integer, Integer> sta = Measures.labelStatic(prob.y);
		
		
		Parameter param = new Parameter(4, 1000, 0.001);
		RevisedMultiLabel rml = new RevisedMultiLabel(prob, param);
		rml.train();
		
		
		
		
		Problem testprob0 = FileIO.readProblem(new File(testfile0), 1);
		Problem testprob1 = FileIO.readProblem(new File(testfile1), 1);
		Problem testprob2 = FileIO.readProblem(new File(testfile2), 1);
		Problem testprob3 = FileIO.readProblem(new File(testfile3), 1);
		
		Problem testprob = new Problem();
		int numOfSamples = testprob0.l + testprob1.l + testprob2.l + testprob3.l;
		double bias = testprob0.bias;
		int n = prob.n;
		int counter = 0;
		
		testprob.l = numOfSamples;
		testprob.n = n;
		testprob.bias = bias;
		testprob.x = new DataPoint[testprob.l][];
		testprob.y = new int[testprob.l][];
		
		int i;
		for(i = 0; i < testprob0.l; i++) {
			testprob.x[counter] = testprob0.x[i];
			testprob.y[counter] = testprob0.y[i];
			counter++;
		}
		
		for(i = 0; i < testprob1.l; i++) {
			testprob.x[counter] = testprob1.x[i];
			testprob.y[counter] = testprob1.y[i];
			counter++;
		}
		
		for(i = 0; i < testprob2.l; i++) {
			testprob.x[counter] = testprob2.x[i];
			testprob.y[counter] = testprob2.y[i];
			counter++;
		}
		
		for(i = 0; i < testprob3.l; i++) {
			testprob.x[counter] = testprob3.x[i];
			testprob.y[counter] = testprob3.y[i];
			counter++;
		}
System.out.println("num of test samples = " + testprob.l);

		int[][] pre = rml.predict(testprob.x);
		double micro_f1 = Measures.microf1(rml.getLabels(), testprob.y, pre);
		double macfo_f1 = Measures.macrof1(rml.getLabels(), testprob.y, pre);
		System.out.println("Micro-F1 = " + micro_f1 + ", Macro-F1 = " + macfo_f1);
	}

}
