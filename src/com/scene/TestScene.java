package com.scene;

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
import com.tools.Statistic;

public class TestScene {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Parameter param = new Parameter(0.25, 1000, 0.001);
		OneVsAllMultilabel rs = new OneVsAllMultilabel(train, param);
		double[] c = new double[15];
		for(int i = -7; i < 8; i++) {
			c[i + 7] = Math.pow(2, i);
		}
		int[] K = {1, 3, 5, 7, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101};
		for(int i = 0; i < c.length; i++) {
//			double[][] perf = rs.gridSerach(train, param, 5, c[i], K);
//			param.setC(c[i]);
//			rs.crossValidation(train, param, 5);
		}
		

		Problem test = Problem.readProblem(new File(testfile), 1);
		
		DataPoint[][] w = rs.train(train, param);
		
		
		double[][] trainpv = rs.predictValues(w, train.x);
		rs.scale(trainpv);
		
		String[] ul = Statistic.getUniqueLabels(train.y);
		double[][][] tcx = Statistic.getClassVector(trainpv, train.y, ul);
		for(int i = 0; i < tcx.length; i++) {
			ul[i] = ul[i].replace(" ", "_");
			String filename = "F:\\java\\RecursiveRegularizationSVM_1\\scene\\train_" + ul[i] + ".txt";
			Statistic.writeMatrixToFile(tcx[i], filename);
		}
		
		double[][] testpv = rs.predictValues(w, test.x);
		rs.scale(testpv);
		
		ul = Statistic.getUniqueLabels(test.y);
		tcx = Statistic.getClassVector(testpv, test.y, ul);
		for(int i = 0; i < tcx.length; i++) {
			ul[i] = ul[i].replace(" ", "_");
			String filename = "F:\\java\\RecursiveRegularizationSVM_1\\scene\\test_" + ul[i] + ".txt";
			Statistic.writeMatrixToFile(tcx[i], filename);
		}
		
//		int[][] pre = rs.predictNear(trainpv, testpv, train.y, 21);
		int[][] pre = rs.predict(testpv);
		
		double microf1 = Measures.microf1(rs.getUniqueLabels(), test.y, pre);
		double macrof1 = Measures.macrof1(rs.getUniqueLabels(), test.y, pre);
		double hammmingLoss = Measures.averageSymLoss(test.y, pre);
		double zeroneloss = Measures.zeroOneLoss(test.y, pre);
		
		System.out.println("C = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammmingLoss + ", Zero One Loss = " + zeroneloss);

//		String testlabel = "F:\\黄亮\\实验数据\\scene\\测试样本标签.txt";
//		String prelabel = "F:\\黄亮\\实验数据\\scene\\测试样本预测标签SVM_KNN.txt";
//		FileIO.writeLabelToFile(testlabel, test.y);
//		FileIO.writeLabelToFile(prelabel, pre);
	}

}
