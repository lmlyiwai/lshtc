package com.flatSvm;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.structure.Structure;
import com.tools.FileIO;
import com.tools.FileInputOutput;

public class TestRCV1 {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "./RCV1RCV2/rcv1.topics.hierorig.txt";
		String qrelfilename = "./RCV1RCV2/rcv1-v2.topics.qrels";
		
		String[][] result = FileInputOutput.readLabelPairs(filename);
		Map<String, Integer> map = FileInputOutput.getIDLabelPair(result);


		int[][] pc = FileInputOutput.labelPairs(result, map);
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
		
		String trainFile = "./RCV1RCV2/vectors/lyrl2004_vectors_train.dat";
	
		Problem prob = FileIO.readProblem(new File(trainFile), 1);
	
		Map<Integer, int[]> iamap = FileInputOutput.getDocLabel(qrelfilename, map);
		prob.getFileLabels(iamap);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		SVMKNN sk = new SVMKNN(prob, param);				
		sk.setTree(tree);
		
//		double[] C = new double[10];
//		for(int i = -4; i < 6; i++) {
//			C[i + 4] = Math.pow(2, i);
//		}
//		
//		int[] K = {3, 5, 7, 11, 21, 31, 41, 51};
//		
//		for(int i = 0; i < C.length; i++) {
//			param.setC(C[i]);
//			for(int j = 0; j < K.length; j++) {
//				System.out.print("c = " + param.getC() + ", k = " + K[j]);
//				sk.crossValidation(prob, param, 5, K[j]);
//			}
//		}
		
		DataPoint[][] w = sk.trainWithInnerNode(prob, param);
		
		String testfile0 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt0.dat";
		String testfile1 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt1.dat";
		String testfile2 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt2.dat";
		String testfile3 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt3.dat";
		
		double[][] trainSample = sk.transformSamples(prob.x, w, prob.n);
		sk.scale(trainSample);
		
		Problem testprob0 = FileIO.readProblem(new File(testfile0), 1);	
		testprob0.getFileLabels(iamap);
		
		double[][] test0 = sk.transformSamples(testprob0.x, w, prob.n);
		sk.scale(test0);
		int[][] pre0 = sk.predictKnearLabels(trainSample, prob.y, test0, 41);
		
		Problem testprob1 = FileIO.readProblem(new File(testfile1), 1);
		testprob1.getFileLabels(iamap);
		
		double[][] test1 = sk.transformSamples(testprob1.x, w, prob.n);
		sk.scale(test1);
		int[][] pre1 = sk.predictKnearLabels(trainSample, prob.y, test1, 41);
		
		Problem testprob2 = FileIO.readProblem(new File(testfile2), 1);
		testprob2.getFileLabels(iamap);
		
		double[][] test2 = sk.transformSamples(testprob2.x, w, prob.n);
		sk.scale(test2);
		int[][] pre2 = sk.predictKnearLabels(trainSample, prob.y, test2, 41);
		
		Problem testprob3 = FileIO.readProblem(new File(testfile3), 1);
		testprob3.getFileLabels(iamap);
		
		double[][] test3 = sk.transformSamples(testprob3.x, w, prob.n);
		sk.scale(test3);
		int[][] pre3 = sk.predictKnearLabels(trainSample, prob.y, test3, 41);
	
		int[][] pre = new int[pre0.length + pre1.length + pre2.length + pre3.length][];
		int[][] testl = new int[pre0.length + pre1.length + pre2.length + pre3.length][];
		
		int counter = 0;
		for(int i = 0; i < pre0.length; i++) {
			pre[counter] = pre0[i];
			testl[counter] = testprob0.y[i];
			counter++;
		}
		
		for(int i = 0; i < pre1.length; i++) {
			pre[counter] = pre1[i];
			testl[counter] = testprob1.y[i];
			counter++;
		}
		
		for(int i = 0; i < pre2.length; i++) {
			pre[counter] = pre2[i];
			testl[counter] = testprob2.y[i];
			counter++;
		}
		
		for(int i = 0; i < pre3.length; i++) {
			pre[counter] = pre3[i];
			testl[counter] = testprob3.y[i];
			counter++;
		}

		double test_microf1 = Measures.microf1(sk.getUlabels(), testl, pre);
		double test_macrof1 = Measures.macrof1(sk.getUlabels(), testl, pre);
		double test_hamming = Measures.averageSymLoss(testl, pre);
		
		System.out.println("Micro-F1 = " + test_microf1 + ", Macro-F1 = " + test_macrof1 + 
				", Hamming Loss = " + test_hamming);
	}

}
