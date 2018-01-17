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

public class TestKnn {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_tfidf_scale.txt";		
		String validfile = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_tfidf_scale.txt";
		String wfile = "DMOZ_weight1_file.txt";
		
		Problem train = ReadData.readProblem(new File(trainfile), 1);
		Problem valid = ReadData.newReadProblem(new File(validfile), 1);
		train = ReadData.mergeProblem(train, valid);
		Parameter param = new Parameter(1, 1000, 0.001);
		System.out.println("train set size " + train.l);
		
		FlatSVM fs = new FlatSVM(train, param);
		double[][] train_pv = fs.predictValues(train.x, wfile);
//		Sigmoid.sigmoid(train_pv, 1);
		
		System.gc();
		System.out.println("读测试样本");
		String testfile = ""; 
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		double[][] test_pv = fs.predictValues(test.x, wfile);
		
		double[] test_entropy = Sigmoid.entropy(test_pv);
//		Sigmoid.sigmoid(test_pv, 1);
		
		double cut = fs.entropyThreshold(test_entropy, 3488);   //
		System.out.println("开始预测");

		
		int[] k = {5, 9, 50, 100, 300, 500, 700, 1000};
		int[][] testplv = new int[test_pv.length][k.length];

		for(int i = 0; i < test_pv.length; i++) {
			System.out.print("predict " + i + ", ");
			if(test_entropy[i] <= cut) {
				long start = System.currentTimeMillis();
				int pl = fs.predict(test_pv[i]);
				for(int j = 0; j < k.length; j++) {
					testplv[i][j] = pl;
				}
				long end = System.currentTimeMillis();
				System.out.println((end - start) + "ms");
			} else {
				long start = System.currentTimeMillis();
				testplv[i] = fs.predictKnnLabels(train_pv, train.y, test_pv[i], k);
				long end = System.currentTimeMillis();
				System.out.println((end - start) + "ms");
			}
		}
		
		for(int i = 0; i < k.length; i++) {
			
			String testlabel = "DMOZ_knn_test_predict_" + k[i] + ".txt";
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(testlabel)));
			for(int j = 0; j < testplv.length; j++) {
				out.write(testplv[j][i] + "\n");
			}
			out.close();
		}
	}

}
