package com.emotion;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.tools.FileIO;

public class TestBinarySVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\emotion\\emotions-train.svm";
		String testfile = "F:\\DataSets\\emotion\\emotions-test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Parameter param = new Parameter(0.2, 1000, 0.001);
		OneVsAllMultilabel rs = new OneVsAllMultilabel(train, param);
		double[] c = new double[15];
		for(int i = -7; i < 8; i++) {
			c[i + 7] = Math.pow(2, i);
		}
		int[] K = {1, 3, 5, 7, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		for(int i = 0; i < c.length; i++) {
//			param.setC(c[i]);
//			rs.crossValidation(train, param, 5);
		}
		
		
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		DataPoint[][] w = rs.train(train, param);
		
		
		double[][] prev = rs.predictValues(w, test.x);
		int[][] pre = rs.predict(prev);
		
		double microf1 = Measures.microf1(rs.getUniqueLabels(), test.y, pre);
		double macrof1 = Measures.macrof1(rs.getUniqueLabels(), test.y, pre);
		double hammmingLoss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("C = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammmingLoss + ", Zero One Loss = " + zeroneloss);
		
		String testlabel = "F:\\黄亮\\实验数据\\scene\\测试样本标签.txt";
		String prelabel = "F:\\黄亮\\实验数据\\scene\\测试样本预测标签BinarySVM.txt";
		FileIO.writeLabelToFile(testlabel, test.y);
		FileIO.writeLabelToFile(prelabel, pre);

	}

}
