package com.dmoz;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.structure.Structure;

public class TestWithCost {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		/**
		 *  每个样本根据结构给出不同的损失 
		 */
		String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";		
		String validfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_ltc_scale.txt";
		String wfile = "DMOZ_diff_cost_weight_c04.txt";
		
		String sfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\cat_hier.txt";
		Map<Integer, Integer> map = ReadData.readMap(sfile);
		Map<Integer, Integer> pam = ReadData.reverseMap(map);
		Structure tree = ReadData.getStructure(sfile, map);
	
		
		Problem train = ReadData.newReadProblem(new File(trainfile), 1);
		Problem valid = ReadData.newReadProblem(new File(validfile), 1);
		ReadData.transLabels(valid.y, map);
		ReadData.transLabels(train.y, map);
		train = ReadData.mergeProblem(train, valid);
		
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
		String testlabel = "DMOZ_diffCost_test_predict_label_ltc_c04.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(testlabel)));
		for(int i = 0; i < pre.length; i++) {
			out.write(pre[i] + "\n");                       
		}
		out.close();
	}

	/**
	 * @throws InvalidInputDataException 
	 * @throws IOException 
	 * 
	 */
	public static double accuracy(String file, String train0, String train1, String train2,
			Map<Integer, Integer> map, FlatSVM fs) throws IOException, InvalidInputDataException {
		double counter = 0;
		double totle = 0;
		Problem t0 = ReadData.readProblem(new File(train0), 1);
		ReadData.transLabels(t0.y, map);
		totle += t0.l;
		
		int[] pre0 = fs.predictMax(t0.x, file);
		for(int j = 0; j < pre0.length; j++) {
			if(pre0[j] == t0.y[j][0]) {
				counter++;
			}
		}
		
		Problem t1 = ReadData.readProblem(new File(train1), 1);
		ReadData.transLabels(t1.y, map);
		totle += t1.l;
		int[] pre1 = fs.predictMax(t1.x,file);
		for(int j = 0; j < pre1.length; j++) {
			if(pre1[j] == t1.y[j][0]) {
				counter++;
			}
		}
		
		Problem t2 = ReadData.readProblem(new File(train2), 1);
		ReadData.transLabels(t2.y, map);
		totle += t2.l;
		int[] pre2 = fs.predictMax(t2.x, file);
		for(int j = 0; j < pre2.length; j++) {
			if(pre2[j] == t2.y[j][0]) {
				counter++;
			}
		}
		
		System.out.println("Accuray = " + (counter / totle));
		return (counter / totle);
	}
	
}
