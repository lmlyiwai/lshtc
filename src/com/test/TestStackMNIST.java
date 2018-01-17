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
import com.rssvm.StackSVM;
import com.sparseVector.DataPoint;
import com.structure.Structure;
import com.tools.FileIO;
import com.tools.FileInputOutput;

public class TestStackMNIST {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\java\\ProcessMNIST\\train.txt";
		String testfile = "F:\\java\\ProcessMNIST\\test.txt";
		
		Problem prob = Problem.readProblem(new File(filename), 1);
		Parameter param = new Parameter(4, 1000, 0.001);
		OneVsAllMultilabel rs = new OneVsAllMultilabel(prob, param);				
		
		double[] C = new double[15];
		for(int i = -7; i < 8; i++) {
			C[i + 7] = Math.pow(2, i);
		}
		
		int[] K = {1, 3, 5, 7, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		
//		for(int i = 0; i < C.length; i++) {
//			double[][] perf = rs.girdSerach(prob, param, 5, C[i], K);
//		}
		
		double c = Math.pow(2, -6);
		param.setC(c);
		DataPoint[][] w = rs.train(prob, param);
			
		double[][] trainpv = rs.predictValues(w, prob.x);
		rs.scale(trainpv);
		
		Problem testprob = Problem.readProblem(new File(testfile), 1);
		double[][] testpv = rs.predictValues(w, testprob.x);
		rs.scale(testpv);
		
		int[][] pre = rs.predictNear(trainpv, testpv, prob.y, 5);
		
		double zeroneLoss = Measures.zeroOneLoss(testprob.y, pre);
		double hammingloss = Measures.averageSymLoss(testprob.y, pre);
		System.out.println("zero one loss = " + zeroneLoss + ", hamming loss = " + hammingloss);
	}

}
