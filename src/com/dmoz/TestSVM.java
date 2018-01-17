package com.dmoz;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.rssvm.Measures;

public class TestSVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\NLPCC\\nram_word\\train.svm.txt";		
//		String validfile = "F:\\DataSets\\NLPCC\\nram_wordF:\\DataSets\\NLPCC\\nram_word\\";
		double cp = 1.0;
		double cn = 1.0;
		String wfile = "NLPCC_ltc_weight_norm_" + cp + "_" + cn + "_file.txt";
		
		Problem train = ReadData.newReadProblem(new File(trainfile), 1);
//		Problem valid = ReadData.newReadProblem(new File(validfile), 1);
//		train = ReadData.mergeProblem(train, valid);
		Parameter param = new Parameter(1, 1000, 0.001);
		FlatSVM fs = new FlatSVM(train, param);
		fs.train(train, param, wfile, cp, cn);

		System.out.println("读测试样本");
		String testfile = "F:\\DataSets\\NLPCC\\nram_word\\test.svm.txt";
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		System.out.println("开始预测");
		int[] pre = fs.predictMax(test.x, wfile);
		
		String testlabel = "NLPCC_ltc_norm_" + cp + "_" + cn + "_test_predict_label.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");                       
		}
		out.close();
		
		int[][] pl = new int[pre.length][1];
		for(int i = 0; i < pl.length; i++) {
			pl[i][0] = pre[i];
		}
		double acc = Measures.zeroOneLoss(test.y, pl);
		double microf1 = Measures.microf1(fs.getUlabels(), test.y, pl);
		double macrof1 = Measures.macrof1(fs.getUlabels(), test.y, pl);
		System.out.println("Acc = " + acc + ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1);
	}

}
