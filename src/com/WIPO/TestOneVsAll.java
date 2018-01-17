package com.WIPO;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.RecursiveSVM;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestOneVsAll {

	@Test
	public void test() throws Exception {
		String filename = "F:\\DataSets\\WIPO\\wipo_train.hf";
		String trainfile = "F:\\DataSets\\WIPO\\wipo_train";
		String testfile = "F:\\DataSets\\WIPO\\wipo_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
				
		Map<Integer, String> idTostr = ProcessIMCLEF.reverseMap(map);
		
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		Parameter param = new Parameter(1, 2000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		prob.y = ProcessIMCLEF.pathToLeaf(prob.y, leaves);
		
		double[] c = new double[15];
		for(int i = -7; i < 8; i++) {
			c[i + 7] = Math.pow(2, i);
		}
		
		RecursiveSVM rs = new RecursiveSVM(tree, prob, param, 0.001);
		for(int i = 0; i < 15; i++) {
			param.setC(c[i]);
//			rs.newCrossValidation(prob, param, 5);
		}
		
		param.setC(2);
		DataPoint[][] w = rs.train(prob, param, 1);
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		testprob.y = ProcessIMCLEF.pathToLeaf(testprob.y, leaves);
		int[][] pre = rs.predict(w, testprob.x);
		double microf1 = Measures.microf1(rs.getLabels(), testprob.y, pre);
		double macrof1 = Measures.macrof1(rs.getLabels(), testprob.y, pre);
		double zeroneloss = Measures.zeroOneLoss(testprob.y, pre);
		double hammingloss = Measures.averageSymLoss(testprob.y, pre);
		System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammingloss + ", Zero One Loss = " + zeroneloss);
	}

}
