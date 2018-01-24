package com.dmoz;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.structure.Structure;

public class TestWithCostDMOZ2012 {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";		
		String wfile = "DMOZ_diff_cost_weight_c04.txt";
		
		String sfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\cat_hier.txt";
		Map<Integer, Integer> map = ReadData.readDMOZ2012Map(sfile);
		Map<Integer, Integer> pam = ReadData.reverseMap(map);
		Structure tree = ReadData.getDMOZ2012Structure(sfile, map);
	
		
		Problem train = ReadData.newReadProblem(new File(trainfile), 1);
		ReadData.transLabels(train.y, map);
		
		Parameter param = new Parameter(0.4, 1000, 0.001);
		FlatSVM fs = new FlatSVM(train, param);
		fs.setTree(tree);
		fs.trainWithDelta(train, param, wfile);
		
		System.gc();
		System.out.println("读测试样本");
		String testfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test_ltc_scale.txt";
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		System.out.println("开始预测");
		int[] pre = fs.predictMax(test.x, wfile);
		for(int i = 0; i < pre.length; i++) {
			pre[i] = pam.get(pre[i]);
		}
		String testlabel = "DMOZ2012_diffCost_test_predict_label_ltc_c04.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");                       
		}
		out.close();
	}

}
