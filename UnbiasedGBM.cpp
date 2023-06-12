#include <bits/stdc++.h>
#include "omp.h"
using namespace std;

string scoretype = "default";
bool DEBUG = true;
int NJOBS = 1;
string LOSSTOOL="";
double LR = 0.;
int N_EST = 1;
int NOW_EST = 0;
int MAX_SPLIT = 31; // [1 or 31]
double large_leaf_reward = 0.5;
double mono_h = 1;

double calc_g(double y, double fpre){
    if (LOSSTOOL=="logloss"){
        fpre = min(fpre, (double)10.);
        if (y==1) return -1./(exp(fpre)+1);
        if (y==0) return exp(fpre)/(exp(fpre)+1);
    }
    if (LOSSTOOL=="MSE"){
        return -(y-fpre);
    }
    throw;
}
double calc_h(double y, double fpre){
    if (mono_h>=0.5) return 1;
    if (LOSSTOOL=="logloss"){
        fpre = min(fpre, (double)10.);
        return exp(fpre)/(exp(2*fpre) + 2*exp(fpre) + 1);
    }
    if (LOSSTOOL=="MSE"){
        return 1;
    }
    throw;
}

vector<double> operator + (vector<double>a, vector<double>b){
    for (int i=0;i<b.size();++i) a[i]+=b[i];
    return a;
}
vector<double> operator - (vector<double>a, vector<double>b){
    for (int i=0;i<b.size();++i) a[i]-=b[i];
    return a;
}
vector<double> operator * (vector<double>a, double b){
    for (auto &o: a) o*=b;
    return a;
}

class Feature{ public:
    string name;
    string type;
    vector<double>dataf;
    vector<int>datai;
    int size(){
        return type=="cat"? datai.size(): dataf.size();
    }
    double operator[](int i){
        return type=="cat"? datai[i]: dataf[i];
    }
};

class DataFrame{ public:
    map<string,int>stoi;
    vector<Feature>feats;
    
    vector<int>shape(){
        vector<int>v;
        v.push_back(feats.size()? feats[0].size(): 0);
        v.push_back(feats.size());
        return v;
    }
};

vector<int> select(vector<int>&vec, vector<int>&idx){
    vector<int> res; res.resize(idx.size()); res.resize(0);
    for (auto i:idx) res.push_back(vec[i]);
    return res;
}
vector<double> select(vector<double>&vec, vector<int>&idx){
    vector<double> res; res.resize(idx.size()); res.resize(0);
    for (auto i:idx) res.push_back(vec[i]);
    return res;
}
Feature select(Feature &f, vector<int>&idx){
    Feature res; res.name = f.name; res.type = f.type;
    if (res.type=="cat") res.datai = select(f.datai, idx);
    else res.dataf = select(f.dataf, idx);
    return res;
}
DataFrame select(DataFrame &df, vector<int>&idx){
    DataFrame res; res.stoi = df.stoi; res.feats.resize(df.feats.size()); res.feats.resize(0);
    for (auto f:df.feats) res.feats.push_back(select(f, idx));
    return res;
}

class Split{ public:
    string name, type;
    unordered_map<int,double>cat_map;
    double fence, org_avg, avg, avg_l, avg_r, std;
    vector<double>score;
};

vector<double> calc_score(vector<double>&x, vector<double>&l){
    vector<double> res;
    if (scoretype=="advance"){
        {
            double xg=x[1], xhp=x[3], xgp=xg;
            double lg=l[1], lhp=l[3], lgp=lg;
            double rg=xg-lg, rhp=xhp-lhp, rgp=xgp-lgp;
            res.push_back((-xg*xgp/xhp+lg*lgp/lhp+rg*rgp/rhp)/2);
        }
        {
            double xg=(x[0])*1.+x[1], xhp=x[2], xgp=x[0];
            double lg=(l[0])*1.+l[1], lhp=l[2], lgp=l[0];
            double rg=xg-lg, rhp=xhp-lhp, rgp=xgp-lgp;
            res.push_back((-xg*xgp/xhp+lg*lgp/lhp+rg*rgp/rhp)/2);
            res.push_back(res.back());
        }
        return res;
    }
    if (scoretype=="origin"){
        {
            double xg=x[0]+x[1], xh=x[2]+x[3];
            double lg=l[0]+l[1], lh=l[2]+l[3];
            double rg=xg-lg, rh=xh-lh;
            res.push_back((-(xg*xg)/xh+(lg*lg)/lh+(rg*rg)/rh)/2);
            res.push_back(res.back());
            res.push_back(res.back());
        }
        return res;
    }
    if (scoretype=="chaos"){
        {
            double xg=(x[0])*1+x[1], xhp=x[2], xgp=x[0];
            double lg=(l[0])*1+l[1], lhp=l[2], lgp=l[0];
            double rg=xg-lg, rhp=xhp-lhp, rgp=xgp-lgp;
            res.push_back((-xg*xgp/xhp+lg*lgp/lhp+rg*rgp/rhp)/2);
            res.push_back(res.back());
            res.push_back(res.back());
        }
        return res;
    }
    if (scoretype=="fair"){
        {
            double xg=x[1], xhp=x[3], xgp=xg;
            double lg=l[1], lhp=l[3], lgp=lg;
            double rg=xg-lg, rhp=xhp-lhp, rgp=xgp-lgp;
            res.push_back((-xg*xgp/xhp+lg*lgp/lhp+rg*rgp/rhp)/2);
        }
        {
            double xg=x[0], xhp=x[2], xgp=xg;
            double lg=l[0], lhp=l[2], lgp=lg;
            double rg=xg-lg, rhp=xhp-lhp, rgp=xgp-lgp;
            res.push_back((-xg*xgp/xhp+lg*lgp/lhp+rg*rgp/rhp)/2);
            res.push_back(res.back());
        }
        return res;
    }
    cerr<<"unknown scoretype : "<<scoretype<<endl;
    exit(1);
}

void calc_split_numerical(vector<double>&value, vector<double>&g, vector<double>&h, vector<int>&bel,
        Split &split, int min_leaf){
    split.score.resize(0);
    split.score.push_back(-1e10); split.score.push_back(-1e10); split.score.push_back(-1e10);
    vector< pair<double,int> >ls; ls.resize(value.size());
    for (int i=0;i<value.size();++i) ls[i] = {value[i], i};
    sort(ls.begin(), ls.end());
    double las=1e9; vector<double>now; now.resize(9);
    
    vector<double> all_4, now_4;
    all_4.resize(4);
    now_4.resize(4);

    for (int i=0;i<value.size();++i){
        if (bel[i]<=1){
            all_4[0] += g[i]; all_4[2] += h[i];
        } else {
            all_4[1] += g[i]; all_4[3] += h[i];
        }
    }
    
    split.org_avg = split.avg = -(all_4[0]+all_4[1])/(all_4[2]+all_4[3]);
    
    int R = 0;
    for (int iter=value.size()-1;iter>=0;--iter){
        int i=ls[iter].second; double v=ls[iter].first;
        now[bel[i]+3] += h[i];
        now[bel[i]+6] += 1;
        if (min(min(now[6],now[7]),now[8]) >= min_leaf)
            if (min(min(now[3],now[4]),now[5]) >= 1e-3){
                R = iter + 1; break;
            }
    }
    now.resize(0); now.resize(9);

    int bestpot = -1, flag = 0;

    for (int iter=0;iter<R;++iter){
        int i=ls[iter].second; double v=ls[iter].first;
        if (iter>0&&v!=las&&(v-las)>1e-9*max(abs(v),abs(las))){
            if (flag) {
                vector<double>score = calc_score(all_4, now_4);
                int pot = rand();
                if (split.score.size()==0 || score[0] > split.score[0] || score[0] == split.score[0] && pot > bestpot){
                    vector<double>els_4 = all_4 - now_4;
                    split.fence = (v+las)/2;
                    split.score = score;
                    split.avg_l = -(now_4[0]+now_4[1])/(now_4[2]+now_4[3]);
                    split.avg_r = -(els_4[0]+els_4[1])/(els_4[2]+els_4[3]);
                    bestpot = pot;
                }
            }
        }
        las = v;
        if (bel[i]<=1){
            now_4[0] += g[i]; now_4[2] += h[i];
        } else {
            now_4[1] += g[i]; now_4[3] += h[i];
        }
        if (flag==0) {
            now[bel[i]+3] += h[i];
            now[bel[i]+6] += 1;
            if (min(min(now[6],now[7]),now[8]) >= min_leaf)
                if (min(min(now[3],now[4]),now[5]) >= 1e-3){
                    flag = 1;
                }
        }
    }
}

void calc_split(Feature &feat, vector<double>&g, vector<double>&h, vector<int>&bel,
        Split &split, int min_leaf){
    split.name = feat.name; 
    split.type = feat.type;
    int n=feat.size();
    if (feat.type=="cat"){
        unordered_map<int,pair<double,double> >mp;
        pair<double, double>sump;
        for (int i=0;i<n;++i)
            if (bel[i]==0) {
                pair<double, double> p = mp[feat.datai[i]];
                mp[feat.datai[i]] = {p.first+g[i], p.second+h[i]};
                sump = {sump.first+g[i], sump.second+h[i]};
            }
        for (auto o:mp) split.cat_map[o.first] = o.second.first / o.second.second;
        double dft = split.cat_map[-1] = sump.first/sump.second;
        vector<double>value; value.resize(n);
        for (int i=0;i<n;++i){
            if (split.cat_map.count(feat.datai[i])) value[i] = split.cat_map[feat.datai[i]];
            else value[i] = dft;
        }
        calc_split_numerical(value, g, h, bel, split, min_leaf);
    }else{
        pair<double, double>sump;
        for (int i=0;i<n;++i)
            if (bel[i]==0) {
                sump = {sump.first+g[i]*h[i], sump.second+h[i]};
            }
        double dft = split.cat_map[-1] = sump.first/sump.second;
        vector<double>value; value.resize(n);
        for (int i=0;i<n;++i) value[i] = isnan(feat.dataf[i])? dft: feat.dataf[i];
        calc_split_numerical(value, g, h, bel, split, min_leaf);
    }
}

class UnbiasedTree{ public:
    UnbiasedTree *left_c; UnbiasedTree *right_c;
    Split split;
    bool is_leaf;

    UnbiasedTree(){ left_c = right_c = NULL; is_leaf = true; split.score.clear(); }
    ~UnbiasedTree(){
        if (!is_leaf){
            delete left_c;
            delete right_c;
        }
    }

    pair<vector<int>, vector<int> >apply_split(DataFrame &df){
        vector<int> idxl, idxr;
        Feature f=df.feats[df.stoi[split.name]];
        if (f.type=="cat"){
            for (int i=0;i<f.size();++i){
                bool is_left = true;
                if (split.cat_map.count(f.datai[i])) is_left = split.cat_map[f.datai[i]] <= split.fence;
                else is_left = split.cat_map[-1] <= split.fence;
                if (is_left) idxl.push_back(i);
                else idxr.push_back(i);
            }
        } else {
            for (int i=0;i<f.size();++i){
                bool is_left = true;
                if (isnan(f.dataf[i])) is_left = split.cat_map[-1] <= split.fence;
                else is_left = f.dataf[i] <= split.fence;
                if (is_left) idxl.push_back(i);
                else idxr.push_back(i);
            }
        }
        return {idxl, idxr};
    }
    void output(){
        printf("%d ", is_leaf);
        if (is_leaf){
            printf("%s %lf\n",split.name.c_str(), split.score[2]);
        } else {
            printf("%s %lf ",split.name.c_str(), split.score[2]);
            printf("%s %lf %lf %lf %lf {", split.type.c_str(), split.fence, split.avg_l, split.avg_r, split.org_avg);
            for (auto o: split.cat_map){
                printf("%d:%lf,",o.first,o.second);
            }
            printf("}");
            printf("\n");
            left_c -> output();
            right_c -> output();
        }
    }
};

int generate_min_leaf(int n, int m){
    return m;
}

struct FuncInput{
    UnbiasedTree **tree_p;
    DataFrame *df;
    vector<double> *g;
    vector<double> *h;
    vector<int> idx;
    int min_leaf;
    double thresh;
    int dep;
};
struct State{
    UnbiasedTree **tree_x;
    double score;
    FuncInput lc, rc;
};
struct cmp{
    bool operator () (const State &a, const State &b){
        return a.score < b.score;
    }
};

priority_queue<State, vector<State>, cmp> pq;
int now_splits = 0;

void prepare_state(UnbiasedTree **tree_p, DataFrame *p_df, vector<double>*p_g, vector<double>*p_h, vector<int>idx, int min_leaf, double thresh, vector<double>*fpres,  int dep=0) {
    DataFrame *df=p_df; vector<double>*g=p_g; vector<double>*h=p_h;
    auto v_df=select((*p_df),idx);
    auto v_g=select((*p_g),idx);
    auto v_h=select((*p_h),idx);
    if (idx.size()>0) {
        df = &v_df; g = &v_g; h = &v_h;
    }

    vector<int>perm, bel;
    bel.resize((*df).shape()[0]); perm.resize((*df).shape()[0]); perm.resize(0);
    for (int i=0;i<(*df).shape()[0];++i) perm.push_back(i);
    random_shuffle(perm.begin(), perm.end());
    for (int i=0;i<bel.size();++i){
        bel[i] = (int)(perm[i]*3/bel.size());
    }

    UnbiasedTree *tree = new UnbiasedTree();
    *tree_p = tree;

    vector<Split>cands; cands.resize((*df).shape()[1]); int g_min_leaf = generate_min_leaf((*df).shape()[0], min_leaf);
    if (NJOBS<=1){
        for (int i=0;i<(*df).shape()[1];++i){
            calc_split((*df).feats[i], (*g), (*h), bel, cands[i], g_min_leaf);
        }
    } else {
        queue<int>tasks; mutex g_mutex;  Split *pos_cands=&cands.front();
        for (int i=0;i<(*df).shape()[1];++i) tasks.push(i);
        #pragma omp parallel num_threads(NJOBS) firstprivate(df, g, h, bel, pos_cands, g_min_leaf) shared(g_mutex, tasks)
        {
            while (1){
                g_mutex.lock();
                if (tasks.empty()) { g_mutex.unlock(); break; }
                int i=tasks.front(); tasks.pop();
                g_mutex.unlock();
                calc_split((*df).feats[i], (*g), (*h), bel, /*cands[i]*/ *(pos_cands+i), g_min_leaf);
            }
        }
    }

    int n_ok_cands = 0, n_numerical = 0;
    for (auto c:cands){
        if (c.type=="num") n_numerical += 1;
        if (c.score[0]>-1e9) n_ok_cands += 1;
    }
    sort(cands.begin(), cands.end(), [](const Split &a, const Split &b){ return a.score[1]>b.score[1]; });
    tree->is_leaf = n_ok_cands <= n_numerical/2.;
    if (cands[0].score[2]<thresh) tree->is_leaf = true;
    if (now_splits >= MAX_SPLIT) tree->is_leaf = true;
    tree->split = cands[0];
    
    if (tree->is_leaf) {
        tree->left_c = NULL; tree->right_c = NULL; return;
    }

    pair<vector<int>,vector<int> > pvv = tree->apply_split((*df));

    tree->split.avg_l -= tree->split.avg;
    tree->split.avg_r -= tree->split.avg;

    vector<int>idxl, idxr;
    if (idx.size()){
        for (auto o:pvv.first) idxl.push_back(idx[o]);
        for (auto o:pvv.second) idxr.push_back(idx[o]);
    }else{
        idxl=pvv.first; idxr=pvv.second;
    }

    if (dep==0) {
        for (auto &o:(*fpres)) o += tree->split.avg * LR;
    }
    for (auto i:idxl) (*fpres)[i] += tree->split.avg_l*LR;
    for (auto i:idxr) (*fpres)[i] += tree->split.avg_r*LR;

    if (finite(tree->split.avg_l)*finite(tree->split.avg_r)==0){
        cerr<<tree->split.avg_l<<" "<<tree->split.avg_r<<"  tree split  "<<tree->split.name<<" "<<tree->split.fence<<" "<<tree->split.score[2]<<endl;
        cerr<<" -> "<<pvv.first.size()<<" "<<pvv.second.size()<<endl;
        cerr<<"   L :"; for (auto i:pvv.first) cerr<<" ("<<(*g)[i]<<","<<(*h)[i]<<")"; cerr<<endl;
        cerr<<"   R :"; for (auto i:pvv.second) cerr<<" ("<<(*g)[i]<<","<<(*h)[i]<<")"; cerr<<endl;
    }

    tree->is_leaf = true; tree->left_c = NULL; tree->right_c = NULL;
    pq.push(
        (State){
            tree_p, /*tree->split.score[2]*/
             (tree->split.score[2])*pow((*df).shape()[0], large_leaf_reward),
            (FuncInput){
                &(tree->left_c), p_df, p_g, p_h, idxl, min_leaf, thresh, dep+1
            },
            (FuncInput){
                &(tree->right_c), p_df, p_g, p_h, idxr, min_leaf, thresh, dep+1
            }
        }
    );
}


void buildUnbiasedTree(UnbiasedTree **tree_p, DataFrame &df, vector<double>g, vector<double>h,
    int min_leaf, double thresh, vector<double>&fpres, int dep=0){
    now_splits = 0;
    while (pq.size()) pq.pop();
    vector<int>nil;
    prepare_state(tree_p, &df, &g, &h, nil, min_leaf, thresh, &fpres, dep);
    while (now_splits < MAX_SPLIT && pq.size()) {
        auto t = pq.top(); pq.pop();
        now_splits += 1;
        (*t.tree_x) -> is_leaf = false;
        prepare_state(t.lc.tree_p, t.lc.df, t.lc.g, t.lc.h, t.lc.idx, t.lc.min_leaf, t.lc.thresh, &fpres, t.lc.dep);
        prepare_state(t.rc.tree_p, t.rc.df, t.rc.g, t.rc.h, t.rc.idx, t.rc.min_leaf, t.rc.thresh, &fpres, t.rc.dep);
    }
}

vector<double> predictUnbiasedTree(UnbiasedTree *tree, DataFrame df, int dep=0){
    vector<double> res; res.resize(df.shape()[0]);
    if (df.shape()[0]==0) return res;
    if (tree->is_leaf) return res;
    pair<vector<int>,vector<int> > pvv = tree->apply_split(df);
    vector<double> lres = predictUnbiasedTree(tree->left_c, select(df, pvv.first), dep+1);
    for (auto &o: lres) o += tree->split.avg_l;
    vector<double> rres = predictUnbiasedTree(tree->right_c, select(df, pvv.second), dep+1);
    for (auto &o: rres) o += tree->split.avg_r;
    if (finite(tree->split.avg_l)*finite(tree->split.avg_r)==0) {
        cerr<<"G in l r  "<<tree->split.avg_l<<" "<<tree->split.avg_r<<endl;
    }
    for (int i=0;i<pvv.first.size();++i) res[pvv.first[i]] = lres[i];
    for (int i=0;i<pvv.second.size();++i) res[pvv.second[i]] = rres[i];
    if (dep==0) for (auto &o:res) o += tree->split.org_avg;
    return res;
}

class UnbiasedBoost{ public:
    vector<UnbiasedTree*>trees;
    string losstool;
    int n_est, min_leaf;
    double thresh, lr;
    UnbiasedBoost(string losstool, int n_est, int min_leaf, double thresh, double lr):
        losstool(losstool), n_est(n_est), min_leaf(min_leaf), thresh(thresh), lr(lr) {  
    }
    ~UnbiasedBoost() {
        for (auto t:trees) delete t;
    }

    vector<double> predict(DataFrame df){
        vector<double>res; res.resize(df.shape()[0]);
        for (auto t:trees){
            res = res + predictUnbiasedTree(t, df)*lr;
        }
        return res;
    }

    void fit(DataFrame df, Feature label, int n_jobs){
        vector<double>fpres = predict(df);
        vector<int>bel; bel.resize(df.shape()[0]);
        for (int _=0;_<n_est;++_){
            if (DEBUG) cerr<<"now at : "<<_<<endl;
            NOW_EST = _;

            vector<double>g, h;
            g.resize(df.shape()[0]); h.resize(df.shape()[0]);
            for (int i=0;i<g.size();++i){
                g[i] = calc_g(label[i], fpres[i]);
                h[i] = calc_h(label[i], fpres[i]);
            }

            trees.resize(trees.size()+1);
            buildUnbiasedTree(&trees[trees.size()-1], df, g, h, min_leaf, thresh, fpres);
        }
    }
    void output(){
        int idx = 0;
        for (auto t:trees){
            t->output();
        }
    }
};


Feature read_feat(){
    Feature f; int m; cin>>f.name>>f.type>>m;
    if (f.type=="cat"){
        f.datai.resize(m); for (auto &o: f.datai) scanf("%d",&o);
    }else{
        f.dataf.resize(m); for (auto &o: f.dataf) scanf("%lf",&o);
    }
    return f;
}

DataFrame read_data(){
    int n; scanf("%d",&n);
    DataFrame df;
    for (int i=0;i<n;++i){
        Feature f = read_feat();
        df.stoi[f.name] = df.feats.size();
        df.feats.push_back(f);
    }
    return df;
}

int main(int argc, char *argv[]) {
    int n_jobs=1, seed=0;
    if (argc>=2){
        sscanf(argv[1],"%d",&n_jobs);
        NJOBS = n_jobs;
    }
    if (argc>=3){
        sscanf(argv[2],"%d",&seed);
        srand(seed);
    }
    if (argc>=4){
        scoretype = argv[3];
    }
    if (argc>=6){
        sscanf(argv[4],"%lf",&large_leaf_reward);
        sscanf(argv[5],"%lf",&mono_h);
    }
    if (argc>=7){
        sscanf(argv[6],"%d",&MAX_SPLIT);
    }
    DEBUG = seed==0;
    
    string task, losstool; int n_est, min_leaf; double thresh, lr;
    cin>>task>>losstool>>n_est>>min_leaf>>thresh>>lr;
    LOSSTOOL = losstool;
    LR = lr;
    N_EST = n_est;

    if (DEBUG) cerr<<task<<' '<<losstool<<' '<<n_est<<' '<<min_leaf<<' '<<thresh<<' '<<lr<<endl;

    DataFrame df=read_data();
    Feature label=read_feat();

    int hasvalid = 0; DataFrame df_valid;
    int hastest = 0; DataFrame df_test;
    scanf("%d",&hasvalid);
    if (hasvalid) {
        df_valid = read_data();
    }
    scanf("%d",&hastest);
    if (hastest) {
        df_test = read_data();
    }

    if (DEBUG) cerr<<"  df "<<df.shape()[0]<<", "<<df.shape()[1]<<endl;
    
    UnbiasedBoost boost = UnbiasedBoost(losstool, n_est, min_leaf, thresh, lr);
    boost.fit(df, label, n_jobs);
    boost.output();

    set<int>S;
    if (large_leaf_reward<-1){
        for (int i=1;i<=n_est;++i) S.insert(i);
    } else {
        S.insert(n_est);
    }

    if (hasvalid) {
        vector<double>pred; pred.resize(df_valid.shape()[0]);
        int iter = 0;
        for (auto t:boost.trees){
            pred = pred + predictUnbiasedTree(t, df_valid)*boost.lr;
            iter += 1;
            if (S.count(iter)){
                printf("%d\n",iter);
                printf("[");
                for (int i=0;i<pred.size();++i){
                    printf("%lf,",pred[i]);
                    if (i%500==0) printf("\n");
                }
                printf("]\n");
            }
        }
    }

    if (hastest) {
        vector<double>pred; pred.resize(df_test.shape()[0]);
        int iter = 0;
        for (auto t:boost.trees){
            pred = pred + predictUnbiasedTree(t, df_test)*boost.lr;
            iter += 1;
            if (S.count(iter)){
                printf("%d\n",iter);
                printf("[");
                for (int i=0;i<pred.size();++i){
                    printf("%lf,",pred[i]);
                    if (i%500==0) printf("\n");
                }
                printf("]\n");
            }
        }
    }
}