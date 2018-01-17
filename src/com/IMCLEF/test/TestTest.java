package com.IMCLEF.test;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestTest {

	@Test
	public void test() throws Exception {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\test\\hie.txt";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\test\\train.txt";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
				
		Map<Integer, String> idTostr = ProcessIMCLEF.reverseMap(map);
		
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		Parameter param = new Parameter(1.0, 1000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		prob.y = ProcessIMCLEF.pathToLeaf(prob.y, leaves);
		FlatSVM fs = new FlatSVM(prob, param);
		fs.setTree(tree);
		
		DataPoint[][] fw = fs.getFirstWeight(prob, param, tree);
		DataPoint[][] sw = fs.getSecondWeight(prob, 1.0, 0.001, tree, fw, prob.n, true, 1000);
		int[][] pl = fs.secondLevelPredictMax(sw, prob.x);
		
		int[][] firstPl = fs.secondLevelPredictMax(fw, prob.x);
		double fmif1 = Measures.microf1(fs.getUlabels(), prob.y, firstPl);
		double fmaf1 = Measures.macrof1(fs.getUlabels(), prob.y, firstPl);
		System.out.println("First Layer, Micro-F1 = " + fmif1 + ", Macro-F1 = " + fmaf1);
		
		double microf1 = Measures.microf1(fs.getUlabels(), prob.y, pl);
		double macrof1 = Measures.macrof1(fs.getUlabels(), prob.y, pl);
		System.out.println("Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1);
	}

}
