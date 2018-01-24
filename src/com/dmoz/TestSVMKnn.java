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
import com.tools.Sigmoid;

public class TestSVMKnn {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_tfidf_scale.txt";		
		String validfile = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_tfidf_scale.txt";
		String w1file = "DMOZ_weight1_file.txt";
		
		Problem train = ReadData.readProblem(new File(trainfile), 1);
		Problem valid = ReadData.newReadProblem(new File(validfile), 1);
		train = ReadData.mergeProblem(train, valid);
		Parameter param = new Parameter(1, 1000, 0.001);
		System.out.println("train set size " + train.l);
		
		FlatSVM fs = new FlatSVM(train, param);
		double[][] train_pv = fs.predictValues(train.x, w1file);
		Sigmoid.sigmoid(train_pv, 1);
		fs.getClassCenter(train_pv, train.y);
		
		System.gc();
		System.out.println("读测试样本");
		String testfile = "";
		String testlabel = "DMOZ_center_test_predict.txt";
		
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		double[][] test_pv = fs.predictValues(test.x, w1file);
		Sigmoid.sigmoid(test_pv, 1);
		System.out.println("开始预测");
		int[] testplv = fs.predict(test_pv);
		for(int i = 0; i < testplv.length; i++) {
			out.write(testplv[i] + "\n");
		}
		out.close();
	}

}
