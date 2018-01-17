package com.test;

import java.io.File;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.RecursiveSVM;
import com.sparseVector.DataPoint;
import com.structure.Structure;
import com.tools.FileIO;
import com.tools.FileInputOutput;

public class TestRCV1 {

	@Test
	public void test() throws Exception {
		String filename = "./RCV1RCV2/rcv1.topics.hierorig.txt";
		String qrelfilename = "./RCV1RCV2/rcv1-v2.topics.qrels";
		
		String[][] result = FileInputOutput.readLabelPairs(filename);
		Map<String, Integer> map = FileInputOutput.getIDLabelPair(result);


		int[][] pc = FileInputOutput.labelPairs(result, map);
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
		tree.extendStructure();
		
		String trainFile = "./RCV1RCV2/vectors/lyrl2004_vectors_train.dat";
		String testfile0 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt0.dat";
		String testfile1 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt1.dat";
		String testfile2 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt2.dat";
		String testfile3 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt3.dat";
	
		Problem prob = FileIO.readProblem(new File(trainFile), 1);
	
		Map<Integer, int[]> iamap = FileInputOutput.getDocLabel(qrelfilename, map);
		prob.getFileLabels(iamap);
		prob.y = FileInputOutput.transLabels(prob.y, tree, tree.getInnerToAdd());
		
		double[] c = new double[15];
		for(int i = 0; i < c.length; i++) {
			c[i] = Math.pow(2, i-7);
		}
		
		Parameter param = new Parameter(0.25, 1000, 0.001);
		RecursiveSVM rs = new RecursiveSVM(tree, prob, param, 0.001);
		
		
		for(int i = 0; i < c.length; i++) {
//			param.setC(c[i]);
//			rs.newCrossValidation(prob, param, 5);
		}
		
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
		testprob.getFileLabels(iamap);
		testprob.y = FileInputOutput.transLabels(testprob.y, tree, tree.getInnerToAdd());
		
		DataPoint[][] w = rs.train(prob, param, 1);
		
		int[][] pre = rs.predict(w, testprob.x);
		double microf1 = Measures.microf1(rs.getLabels(), testprob.y, pre);
		double macrof1 = Measures.macrof1(rs.getLabels(), testprob.y, pre);
		double haml = Measures.averageSymLoss(testprob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + 
				", Macro-F1 = " + macrof1 + ", Hamming Loss = " + haml);
	}

}
