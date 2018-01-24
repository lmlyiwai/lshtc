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

public class TestSecondLayer {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		/**
		 *  每个样本根据结构给出不同的损失 
		 */
		String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";		
		String validfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_ltc_scale.txt";
		String wfile1 = "DMOZ_inner_node_first_layer_weight.txt";
		String wfile2 = "DMOZ_inner_node_second_layer_weight.txt";
		
		String sfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\cat_hier.txt";
		Map<Integer, Integer> map = ReadData.readMap(sfile);
		Map<Integer, Integer> pam = ReadData.reverseMap(map);
		Structure tree = ReadData.getStructure(sfile, map);
	
		
		Problem train = ReadData.readProblem(new File(trainfile), 1);
		Problem valid = ReadData.readProblem(new File(validfile), 1);
		ReadData.transLabels(valid.y, map);
		ReadData.transLabels(train.y, map);
		train = ReadData.mergeProblem(train, valid);
		
		Parameter param1 = new Parameter(1, 1000, 0.001);
		Parameter param2 = new Parameter(5, 1000, 0.001);
		
		FlatSVM fs = new FlatSVM(train, param1);
		fs.setTree(tree);

		fs.stackSecondLayer(train, param1, param2, tree, wfile1, wfile2);
		
		System.gc();
		System.out.println("读测试样本");
		String testfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test_ltc_scale.txt";
		Problem test = ReadData.newReadProblem(new File(testfile), 1);
		System.out.println("开始预测");
		
		int[] pre = fs.stackPredict(test.x, wfile1, wfile2, tree);
		
		for(int i = 0; i < pre.length; i++) {
			pre[i] = pam.get(pre[i]);
		}
		String testlabel = "dmoz_c1_" + param1.getC() + "_c2_" + param2.getC() + "_test_label.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");                       
		}
		out.close();
	}

}
