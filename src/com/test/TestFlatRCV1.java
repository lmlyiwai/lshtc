package com.test;

import java.io.File;
import java.util.Arrays;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.tools.Contain;
import com.tools.FileIO;
import com.tools.FileInputOutput;

public class TestFlatRCV1 {

	@Test
	public void test() throws Exception, InvalidInputDataException {
		
		String trainFile = "F:\\DataSets\\rcv1\\rcv1_topics_train.svm";
		String testfile0 = "F:\\DataSets\\rcv1\\rcv1_topics_test_0.svm";
		String testfile1 = "F:\\DataSets\\rcv1\\rcv1_topics_test_1.svm";
		String testfile2 = "F:\\DataSets\\rcv1\\rcv1_topics_test_2.svm";
		String testfile3 = "F:\\DataSets\\rcv1\\rcv1_topics_test_3.svm";
		
		Problem prob = FileIO.readProblem(new File(trainFile), 1);	
		
		Map<Integer, Integer> sta = Measures.labelStatic(prob.y);
		
		FileInputOutput.writeMapToFile(sta, "rcv1_statics.txt");
		
		Parameter param = new Parameter(1.7, 1000, 0.001);
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
		
		int[] labels = {34,59,93,94,102,21,33,5,4,51,70,79,0,19,98,97,95,40,1,71,2,22,32,35,96,15,14,83,101,55,53,60,68,73,90,7,18,100,26,36,38,37,42,27,9,23,24,45,44,89,13,46,75,6,3,31,30,25,17,58,99,29,11,8,64,65,10,67,62,87,52,74,28,16,76,78,12,54,43,72,91,20,61,82,85,92,88,84,56,39,86,81,63,47,48,41,57,50,69,77,66
		};
		
		Arrays.sort(labels);
		
		DataPoint[][] w = new DataPoint[103][];
		
		int id;
		int[] tl = null;
		int j;
		for(i = 0; i < w.length; i++) {
			id = i;
			if(!Contain.contain(labels, id)) {
				System.out.println("label id " + id);
				continue;
			}
			
			tl = new int[prob.l];
			for(j = 0; j < tl.length; j++) {
				if(Contain.contain(prob.y[j], id)) {
					tl[j] = 1;
				} else {
					tl[j] = -1;
				}
			}
			double[] loss = new double[1];
			w[id] = Linear.train(prob, tl, param, null, loss, null, 0);
		}
		
		double[][] weight = new double[w.length][];
		for(i = 0; i < weight.length; i++) {
			weight[i] = SparseVector.sparseVectorToArray(w[i], prob.n);
		}
		
		int[][] pre = new int[testprob.l][];
		double[] pv = new double[103];
		counter = 0;

		for(i = 0; i < testprob.l; i++) {
			counter = 0;
			for(j = 0; j < 103; j++) {
				pv[j] = SparseVector.innerProduct(weight[j], testprob.x[i]);
				if(pv[j] > 0) {
					counter++;
				}
			}
			
			pre[i] = new int[counter];
			counter = 0;
			for(j = 0; j < pv.length; j++) {
				if(pv[j] > 0) {
					pre[i][counter++] = j;
				}
			}
		}
		
		double microF1 = Measures.microf1(labels, testprob.y, pre);
		double macroF1 = Measures.macrof1(labels, testprob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microF1 + ", Macro-F1 = " + macroF1);

	}
}
