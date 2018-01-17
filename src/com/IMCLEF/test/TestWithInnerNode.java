package com.IMCLEF.test;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestWithInnerNode {

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
		Parameter param = new Parameter(1, 2000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		prob.y = ProcessIMCLEF.pathToLeaf(prob.y, leaves);
		FlatSVM fs = new FlatSVM(prob, param);
		fs.setTree(tree);
		
		
		Parameter param1 = new Parameter(3, 1000, 0.001);
		Parameter param2 = new Parameter(0.5, 1000, 0.001);
		
		String wfile1 = "weight1.txt";
		String wfile2 = "weight2.txt";
		
		fs.stack(prob, param1, param2, tree, wfile1, wfile2);
		
		
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		testprob.y = ProcessIMCLEF.pathToLeaf(testprob.y, leaves);
			
		int[] pl = fs.stackPredict(testprob.x, wfile1, wfile2, tree);
		
		int[][] pre = new int[pl.length][1];
		for(int i = 0; i < pre.length; i++) {
			pre[i][0] = pl[i];
		}

		double counter = 0;
		for(int i = 0; i < pre.length; i++) {
			if(pl[i] == testprob.y[i][0]) {
				counter++;
			}
		}
		
		double microf1 = Measures.microf1(tree.getLeaves(), testprob.y, pre);
		double macrof1 = Measures.macrof1(tree.getLeaves(), testprob.y, pre);
		double hamminloss = Measures.averageSymLoss(testprob.y, pre);
		double zeroneloss = 1 - counter / pl.length;
		System.out.println("Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 +
				", Hamming Loss = " + hamminloss + ", 0/1 loss = " + zeroneloss);
	}

}
