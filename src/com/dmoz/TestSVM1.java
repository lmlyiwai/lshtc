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

public class TestSVM1 {

	@Test
	public void test() throws Exception, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\"
				+ "large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";		
		String validfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\"
				+ "large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_ltc_scale.txt";
		double cp = 2.0;
		double cn = 1.0;
		String wfile = "DMOZ_ltc_weight_norm_" + cp + "_" + cn + "_file.txt";
		
		Problem train = ReadData.newReadProblem(new File(trainfile), 1);
		Problem valid = ReadData.newReadProblem(new File(validfile), 1);
		train = ReadData.mergeProblem(train, valid);
		Parameter param = new Parameter(1, 1000, 0.001);
		FlatSVM fs = new FlatSVM(train, param);
		fs.train(train, param, wfile, cp, cn);

		System.out.println("读测试样本");
		String testfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\"
				+ "large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test_ltc_scale.txt";
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		System.out.println("开始预测");
		int[] pre = fs.predictMax(test.x, wfile);
		
		String testlabel = "DMOZ_ltc_norm_" + cp + "_" + cn + "_test_predict_label.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");                       
		}
		out.close();
	}

}
