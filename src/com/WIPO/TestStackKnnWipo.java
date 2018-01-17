package com.WIPO;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestStackKnnWipo {

	@Test
	public void test() throws IOException, InvalidInputDataException {
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
		int[] K = {1, 3, 5, 7, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		
		OneVsAllMultilabel rs = new OneVsAllMultilabel(prob, param);
		for(int i = 0; i < 15; i++) {
			double tc = c[i];
//			rs.girdSerach(prob, param, 5, tc, K);
		}
		
		param.setC(0.5);
		DataPoint[][] w = rs.train(prob, param);
		
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		testprob.y = ProcessIMCLEF.pathToLeaf(testprob.y, leaves);
		
		double[][] trainpv = rs.predictValues(w, prob.x);
		rs.scale(trainpv);
		
		double[][] testpv = rs.predictValues(w, testprob.x);
		rs.scale(testpv);
		
		int[][] pre = rs.predictNear(trainpv, testpv, prob.y, 11);
		
		double microf1 = Measures.microf1(rs.getUniqueLabels(), testprob.y, pre);
		double macrof1 = Measures.macrof1(rs.getUniqueLabels(), testprob.y, pre);
		double hammmingLoss = Measures.averageSymLoss(testprob.y, pre);
		double zeroneloss = Measures.zeroOneLoss(testprob.y, pre);
		
		System.out.println("C = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammmingLoss + ", Zero One Loss = " + zeroneloss);

	}

}
