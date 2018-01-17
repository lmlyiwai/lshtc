package com.test;

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

public class TestCrossValidation {

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
		String testfile0 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt0.dat";
		String testfile1 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt1.dat";
		String testfile2 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt2.dat";
		String testfile3 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt3.dat";
	
		Problem prob = FileIO.readProblem(new File(trainFile), 1);
	
		Map<Integer, int[]> iamap = FileInputOutput.getDocLabel(qrelfilename, map);
		prob.getFileLabels(iamap);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		OneVsAllMultilabel rs = new OneVsAllMultilabel(prob, param);				
		
		double[] C = new double[15];
		for(int i = -7; i < 8; i++) {
			C[i + 7] = Math.pow(2, i);
		}
		
		int[] K = {7};
		double[] exp = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
				15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300, 500, 1000};
		for(int i = 0; i < exp.length; i++) {
			System.out.println("exp = " + exp[i]);
			double[][] perf = rs.newGridSerach(prob, param, 5, 2, K, exp[i]);
		}
	}

}
