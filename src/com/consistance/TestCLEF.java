package com.consistance;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.RecursiveSVM;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestCLEF {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a_train.hf";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a_train";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
				
		Map<Integer, String> idTostr = ProcessIMCLEF.reverseMap(map);
		
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		Parameter param = new Parameter(1, 1, 1000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		
		double[] c = new double[15];
		double[] c1 = new double[15];
		for(int i = -7; i < 8; i++) {
			c[i + 7] = Math.pow(2, i);
			c1[i + 7] = Math.pow(2, i);
		}
		
		ConsistentSVM cs = new ConsistentSVM(prob, param, tree);
		for(int i = 0; i < 15; i++) {
			for(int j = 0; j < 15; j++) {
				param.setC(c[i]);
				param.setC1(c1[j]);
				double[] perf = cs.crossValidation(prob, param, 5);
			}
		}
		
		cs.train(prob, param);
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		
		int[][] pre = cs.predict(testprob.x);
		double microf1 = Measures.microf1(cs.getLabels(), testprob.y, pre);
		double macrof1 = Measures.macrof1(cs.getLabels(), testprob.y, pre);
		double zeroneloss = Measures.zeroOneLoss(testprob.y, pre);
		double hammingloss = Measures.averageSymLoss(testprob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammingloss + ", Zero One Loss = " + zeroneloss);
	}

}
