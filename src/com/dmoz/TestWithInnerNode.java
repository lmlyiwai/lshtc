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

public class TestWithInnerNode {

    @Test
    public void test() throws IOException, InvalidInputDataException {
        /**
         *  ????????????????????????
         */
        String trainfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\train_ltc_scale.txt";
        String validfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\validation_ltc_scale.txt";
        String wfile = "DMOZ_inner_node_weight_c1.txt";

        String sfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\cat_hier.txt";
        Map<Integer, Integer> map = ReadData.readMap(sfile);
        Map<Integer, Integer> pam = ReadData.reverseMap(map);
        Structure tree = ReadData.getStructure(sfile, map);

        int[] level1 = tree.getChildren(tree.getRoot());
        System.out.println("?????" + level1.length);

        for (int i = 0; i < level1.length; i++) {
            int[] des = tree.getDescendent(level1[i]);
            System.out.println(i + " - " + des.length);
        }
        Problem train = ReadData.readProblem(new File(trainfile), 1);
        Problem valid = ReadData.readProblem(new File(validfile), 1);
        ReadData.transLabels(valid.y, map);
        ReadData.transLabels(train.y, map);
        train = ReadData.mergeProblem(train, valid);

        Parameter param = new Parameter(1, 1000, 0.001);
        FlatSVM fs = new FlatSVM(train, param);
        fs.setTree(tree);

        fs.trainWithInnerNode(train, param, wfile);

        System.gc();
        System.out.println("??????????");
        String testfile = "F:\\DataSets\\Dmoz\\DMOZ2011\\large_lshtc_dataset\\large_lshtc_dataset\\Task1_Train_CrawlData_Test_CrawlData\\test_ltc_scale.txt";
        Problem test = ReadData.newReadProblem(new File(testfile), 1);
        System.out.println("??????");
//		int[] pre = fs.predictMax(test.x, wfile);

        int[] pre = fs.predictWithInnerNode(test.x, wfile, tree, train.n);
        for (int i = 0; i < pre.length; i++) {
            pre[i] = pam.get(pre[i]);
        }
        String testlabel = "DMOZ_with_inner_node_test_predict_label_ltc_c1.txt";
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(testlabel)));
        for (int i = 0; i < pre.length; i++) {
            out.write(pre[i] + "\n");
        }
        out.close();
    }

}
