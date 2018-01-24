package com.BinarySVM;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.Contain;
import com.tools.Matrix;
import com.tools.ProcessProblem;
import com.tools.RandomSequence;
import com.tools.Sort;

public class BinarySVM {
    private DataPoint[][] w;
    private int[] ulabels;
    private Structure tree;

    public BinarySVM(Structure tree) {
        this.tree = tree;
    }

    public DataPoint[][] train(Problem train, Parameter param) {
        DataPoint[][] weight = new DataPoint[this.ulabels.length][];
        int label;
        for (int i = 0; i < this.ulabels.length; i++) {
            label = this.ulabels[i];
            int[] ty = getLabels(train.y, label);
            double[] margin = getLoss(train.y, label);
            weight[i] = Linear.train(train, ty, param, margin);
        }
        return weight;
    }

    public DataPoint[][] newTrain(Problem train, Parameter param) {
        DataPoint[][] weight = new DataPoint[this.ulabels.length][];
        int label;
        for (int i = 0; i < this.ulabels.length; i++) {
            label = this.ulabels[i];
            double[] margin = getLoss1(train.y, label);
            weight[i] = Linear.train(train, param, margin);
        }
        return weight;
    }

    public int[][] crossValidation(Problem prob, Parameter param, int n_fold) {
        int n = prob.l;

        int[][] pre = new int[n][];

        int[] index = RandomSequence.randomSequence(n);

        int segLength = n / n_fold;

        int vbegin = 0;
        int vend = 0;

        int[] validIndex = null;
        int[] trainIndex = null;
        int counter = 0;
        for (int i = 0; i < n_fold; i++) {
            vbegin = i * segLength;
            vend = i * segLength + segLength;

            validIndex = new int[vend - vbegin];
            trainIndex = new int[n - validIndex.length];

            counter = 0;
            for (int j = vbegin; j < vend; j++) {
                validIndex[counter++] = index[j];
            }

            counter = 0;
            for (int j = 0; j < vbegin; j++) {
                trainIndex[counter++] = index[j];
            }
            for (int j = vend; j < n; j++) {
                trainIndex[counter++] = index[j];
            }

            Problem train = new Problem();
            train.l = trainIndex.length;
            train.n = prob.n;
            train.bias = prob.bias;
            train.x = new DataPoint[trainIndex.length][];
            train.y = new int[trainIndex.length][];

            counter = 0;
            for (int j = 0; j < trainIndex.length; j++) {
                train.x[counter] = prob.x[trainIndex[j]];
                train.y[counter] = prob.y[trainIndex[j]];
                counter++;
            }

            Problem valid = new Problem();
            valid.l = validIndex.length;
            valid.n = prob.n;
            valid.bias = prob.bias;
            valid.x = new DataPoint[validIndex.length][];
            valid.y = new int[validIndex.length][];

            counter = 0;
            for (int j = 0; j < validIndex.length; j++) {
                valid.x[counter] = prob.x[validIndex[j]];
                valid.y[counter] = prob.y[validIndex[j]];
                counter++;
            }

            DataPoint[][] w = train(train, param);

            int[][] predictLabel = predict(w, valid.x);

            for (int j = 0; j < validIndex.length; j++) {
                pre[validIndex[j]] = predictLabel[j];
            }
        }
        double microf1 = Measures.microf1(this.ulabels, prob.y, pre);
        double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre);
        double hammingloss = Measures.averageSymLoss(prob.y, pre);
        System.out.println("c = " + param.getC() + ", Micro-F1 = " + microf1 + ", Macro-F1 = "
                + macrof1 + ", Hamming Loss = " + hammingloss);
        return pre;
    }

    public void crossValidation(Problem prob, Parameter param, int n_fold, int[] k) {
        this.ulabels = ProcessProblem.getUniqueLabels(prob.y);

        int n = prob.l;

        int[][][] pre = new int[k.length][n][];

        int[] index = RandomSequence.randomSequence(n);

        int segLength = n / n_fold;

        int vbegin = 0;
        int vend = 0;

        int[] validIndex = null;
        int[] trainIndex = null;
        int counter = 0;
        for (int i = 0; i < n_fold; i++) {
            vbegin = i * segLength;
            vend = i * segLength + segLength;

            validIndex = new int[vend - vbegin];
            trainIndex = new int[n - validIndex.length];

            counter = 0;
            for (int j = vbegin; j < vend; j++) {
                validIndex[counter++] = index[j];
            }

            counter = 0;
            for (int j = 0; j < vbegin; j++) {
                trainIndex[counter++] = index[j];
            }
            for (int j = vend; j < n; j++) {
                trainIndex[counter++] = index[j];
            }

            Problem train = new Problem();
            train.l = trainIndex.length;
            train.n = prob.n;
            train.bias = prob.bias;
            train.x = new DataPoint[trainIndex.length][];
            train.y = new int[trainIndex.length][];

            counter = 0;
            for (int j = 0; j < trainIndex.length; j++) {
                train.x[counter] = prob.x[trainIndex[j]];
                train.y[counter] = prob.y[trainIndex[j]];
                counter++;
            }

            Problem valid = new Problem();
            valid.l = validIndex.length;
            valid.n = prob.n;
            valid.bias = prob.bias;
            valid.x = new DataPoint[validIndex.length][];
            valid.y = new int[validIndex.length][];

            counter = 0;
            for (int j = 0; j < validIndex.length; j++) {
                valid.x[counter] = prob.x[validIndex[j]];
                valid.y[counter] = prob.y[validIndex[j]];
                counter++;
            }

            DataPoint[][] w = train(train, param);
            double[][] trainpv = predictValues(w, train.x);
            scale(trainpv);

            double[][] validpv = predictValues(w, valid.x);
            scale(validpv);

            int[][][] temppre = new int[k.length][valid.l][];
            for (int h = 0; h < validpv.length; h++) {
                double[] dis = distance(trainpv, validpv[h]);
                int[] ind = Sort.getIndexBeforeSort(dis);
                for (int m = 0; m < k.length; m++) {
                    int[][] ty = getFirstKY(ind, train.y, k[m]);
                    int[] tpy = voteLabel(ty);
                    temppre[m][h] = tpy;
                }
            }

            for (int h = 0; h < valid.l; h++) {
                for (int m = 0; m < k.length; m++) {
                    pre[m][validIndex[h]] = temppre[m][h];
                }
            }
        }
        double[][] performance = new double[k.length][2];
        for (int i = 0; i < k.length; i++) {
            double microf1 = Measures.microf1(this.ulabels, prob.y, pre[i]);
            double macrof1 = Measures.macrof1(this.ulabels, prob.y, pre[i]);
            performance[i][0] = microf1;
            performance[i][1] = macrof1;
            double hammingLoss = Measures.averageSymLoss(prob.y, pre[i]);
            double zeroneloss = Measures.zeroOneLoss(prob.y, pre[i]);
            System.out.println("C = " + param.getC() + ", K = " + k[i] +
                    ", Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 +
                    ", Hamming Loss = " + hammingLoss +
                    ", zero one loss = " + zeroneloss);
        }
    }

    /**
     * 距离 ,最大距离
     */
    public double[] getLoss(int[][] y, int label) {
        double MaxDis = this.tree.getMaxDistance();
        double[] margin = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            if (Contain.contain(y[i], label)) {
                margin[i] = MaxDis;
            } else {
                margin[i] = maxDistance(y[i], label);
            }
        }
        return margin;
    }

    /**
     * 距离 ,最大距离
     */
    public double[] getLoss1(int[][] y, int label) {
        double[] margin = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            if (Contain.contain(y[i], label)) {
                margin[i] = 1;
            } else {
                margin[i] = maxDistance(y[i], label);
            }
        }
        return margin;
    }

    /**
     *
     */
    public double maxDistance(int[] y, int label) {
        double distance = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < y.length; i++) {
            double td = this.tree.getDistance(y[i], label);
            if (td > distance) {
                distance = td;
            }
        }
        return distance;
    }

    /**
     *
     */
    public int[] getLabels(int[][] y, int label) {
        int[] labels = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            if (Contain.contain(y[i], label)) {
                labels[i] = 1;
            } else {
                labels[i] = -1;
            }
        }
        return labels;
    }

    /**
     *
     */
    public int[][] predict(DataPoint[][] weight, DataPoint[][] test) {
        double[][] predictValue = new double[weight.length][test.length];
        for (int i = 0; i < weight.length; i++) {
            DataPoint[] tw = weight[i];
            for (int j = 0; j < test.length; j++) {
                predictValue[i][j] = SparseVector.innerProduct(tw, test[j]);
            }
        }

        int[][] predictLabel = new int[test.length][];
        for (int i = 0; i < test.length; i++) {
            double[] pv = getMatrixColumn(predictValue, i);
            int[] pl = getPredict(pv);
            predictLabel[i] = pl;
        }
        return predictLabel;
    }

    /**
     *
     */
    public double[][] predictValues(DataPoint[][] w, DataPoint[][] test) {
        double[][] predictValue = new double[w.length][test.length];
        for (int i = 0; i < w.length; i++) {
            DataPoint[] tw = w[i];
            for (int j = 0; j < test.length; j++) {
                predictValue[i][j] = SparseVector.innerProduct(tw, test[j]);
            }
        }

        predictValue = Matrix.trans(predictValue);
        return predictValue;
    }

    /**
     * 获得距离排序
     */
    public int[] getSortIndex(double[][] pv, double[] t) {
        double[] dis = new double[pv.length];
        for (int i = 0; i < pv.length; i++) {
            double[] sub = SparseVector.subVector(pv[i], t);
            double inp = SparseVector.innerProduct(sub, sub);
            double tdis = Math.pow(inp, 0.5);
            dis[i] = tdis;
        }
        int[] ind = Sort.getIndexBeforeSort(dis);
        return ind;
    }

    public int[][] getKLabels(int[][] y, int[] ind, int[] k) {
        int[][] pl = new int[k.length][];
        for (int i = 0; i < k.length; i++) {
            int[][] ty = new int[k[i]][];
            for (int j = 0; j < ty.length; j++) {
                ty[j] = y[ind[j]];
            }
            pl[i] = getLabel(ty);
        }
        return pl;
    }

    /**
     *
     */
    public int[] getLabel(int[][] ty) {
        double n = (double) ty.length / 2;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int key, value;
        for (int i = 0; i < ty.length; i++) {
            for (int j = 0; j < ty[i].length; j++) {
                key = ty[i][j];
                if (map.containsKey(key)) {
                    value = map.get(key);
                    value = value + 1;
                    map.put(key, value);
                } else {
                    map.put(key, 1);
                }
            }
        }

        Set<Integer> set = map.keySet();
        Iterator<Integer> it = set.iterator();
        List<Integer> list = new ArrayList<Integer>();
        while (it.hasNext()) {
            key = it.next();
            value = map.get(key);
            if (value > n) {
                list.add(key);
            }
        }
        int[] result = new int[list.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = list.get(i);
        }
        return result;
    }

    /**
     *
     */
    public double[] getMatrixColumn(double[][] matrix, int col) {
        double[] c = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            c[i] = matrix[i][col];
        }
        return c;
    }

    /**
     *
     */
    public int[] getPredict(double[] pv) {
        int counter = 0;
        for (int i = 0; i < pv.length; i++) {
            if (pv[i] > 0) {
                counter = counter + 1;
            }
        }

        int[] pl = new int[counter];
        counter = 0;
        for (int i = 0; i < pv.length; i++) {
            if (pv[i] > 0) {
                pl[counter] = this.ulabels[i];
                counter = counter + 1;
            }
        }
        return pl;
    }

    /**
     *
     */
    public void getUlabels(int[][] y) {
        Set<Integer> set = new HashSet<Integer>();
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < y[i].length; j++) {
                set.add(y[i][j]);
            }
        }

        int[] labels = new int[set.size()];
        Iterator<Integer> it = set.iterator();
        int counter = 0;
        while (it.hasNext()) {
            labels[counter] = it.next();
            counter = counter + 1;
        }
        this.ulabels = labels;
    }

    public int[] getUlabels() {
        return ulabels;
    }

    public void setUlabels(int[] ulabels) {
        this.ulabels = ulabels;
    }

    public int[][] predictSingleLabel(DataPoint[][] weight, DataPoint[][] test) {
        double[][] predictValue = new double[weight.length][test.length];
        for (int i = 0; i < weight.length; i++) {
            DataPoint[] tw = weight[i];
            for (int j = 0; j < test.length; j++) {
                predictValue[i][j] = SparseVector.innerProduct(tw, test[j]);
            }
        }

        int[][] predictLabel = new int[test.length][1];
        for (int i = 0; i < test.length; i++) {
            double[] pv = getMatrixColumn(predictValue, i);
            int pl = getMax(pv);
            predictLabel[i][0] = pl;
        }
        return predictLabel;
    }

    /**
     *
     */
    public int getMax(double[] pv) {
        int index = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < pv.length; i++) {
            if (pv[i] > max) {
                max = pv[i];
                index = i;
            }
        }

        int label = this.ulabels[index];
        return label;
    }

    /**
     * 向量归一化
     */
    public void scale(double[][] pv) {
        double norm = 0;
        double[] temp = null;
        for (int i = 0; i < pv.length; i++) {
            temp = pv[i];
            norm = SparseVector.innerProduct(temp, temp);
            norm = Math.pow(norm, 0.5);
            for (int j = 0; j < pv[i].length; j++) {
                pv[i][j] = pv[i][j] / norm;
            }
        }
    }

    /**
     *
     */
    public double[] distance(double[][] pv, double[] p) {
        double[] result = new double[pv.length];
        double[] sub = null;
        for (int i = 0; i < result.length; i++) {
            sub = SparseVector.subVector(pv[i], p);
            result[i] = SparseVector.innerProduct(sub, sub);
        }
        return result;
    }

    /**
     *
     */
    public int[][] getFirstKY(int[] index, int[][] y, int k) {
        int[][] result = new int[k][];
        for (int i = 0; i < k; i++) {
            result[i] = y[index[i]];
        }
        return result;
    }

    /**
     *
     */
    public int[] voteLabel(int[][] y) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int key;
        int value;
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < y[i].length; j++) {
                key = y[i][j];
                if (map.containsKey(key)) {
                    value = map.get(key);
                    value = value + 1;
                    map.put(key, value);
                } else {
                    map.put(key, 1);
                }
            }
        }

        List<Integer> list = new ArrayList<Integer>();
        double n = (double) y.length / 2;

        Set<Integer> set = map.keySet();
        Iterator<Integer> it = set.iterator();
        while (it.hasNext()) {
            key = it.next();
            value = map.get(key);
            if (value > n) {
                list.add(key);
            }
        }

        int[] result = new int[list.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = list.get(i);
        }
        return result;
    }

    /**
     *
     */
    public int[][] predictNear(double[][] pv, double[][] testpv, int[][] y, int k) {
        int[][] result = new int[testpv.length][];

        for (int i = 0; i < result.length; i++) {
            result[i] = getNearestLabel(pv, testpv[i], y, k);
        }

        return result;
    }

    /**
     *
     */
    public int[] getNearestLabel(double[][] pv, double[] testpv, int[][] y, int n) {
        double inner = 0;

        double[] sub = null;
        double[] distance = new double[pv.length];
        for (int i = 0; i < pv.length; i++) {

            sub = SparseVector.subVector(pv[i], testpv);

            inner = SparseVector.innerProduct(sub, sub);

            distance[i] = inner;
        }

        int[] index = Sort.getIndexBeforeSort(distance);

        int[][] pre = new int[n][];
        for (int i = 0; i < n; i++) {
            pre[i] = y[index[i]];
        }

        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < pre.length; i++) {
            for (int j = 0; j < pre[i].length; j++) {
                if (map.containsKey(pre[i][j])) {
                    int value = map.get(pre[i][j]);
                    value = value + 1;
                    map.put(pre[i][j], value);
                } else {
                    map.put(pre[i][j], 1);
                }
            }
        }

        Set<Integer> set = map.keySet();
        Iterator<Integer> it = set.iterator();
        int key;
        int value;

        List<Integer> list = new ArrayList<Integer>();
        while (it.hasNext()) {
            key = it.next();
            value = map.get(key);
            if (value > ((double) n / 2)) {
                list.add(key);
            }
        }

        int[] result = new int[list.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = list.get(i);
        }
        return result;
    }
}

