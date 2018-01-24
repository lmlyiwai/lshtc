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

public class TestThresholdDMOZ {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";		
		String validfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_ltc_scale.txt";
		String wfile = "DMOZ_ltc_weight_with_innernode_C1_file.txt";
		
		Problem train = ReadData.readProblem(new File(trainfile), 1);
		Problem valid = ReadData.readProblem(new File(validfile), 1);
		train = ReadData.mergeProblem(train, valid);
		Parameter param = new Parameter(1, 1000, 0.001);
		FlatSVM fs = new FlatSVM(train, param);
//		fs.train(train, param, wfile);
		double[][] tpv = fs.predictValues(train.x, wfile);
		
		double[][] t = fs.getTPbpuV1(tpv, train.y, 1);
		
		System.gc();
		System.out.println("读测试样本");
		String testfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test_ltc_scale.txt";
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		System.out.println("开始预测");
		
		double[][] testpv = fs.predictValues(test.x, wfile);
		int[] pre = fs.predict(testpv, t[0]);
		
		String testlabel = "DMOZ_ltc_C1_test_predict_label.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");                       
		}
		out.close();
	}

}
