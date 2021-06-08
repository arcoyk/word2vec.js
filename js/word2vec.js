// tf.enableDebugMode()

word2vec = {};
function init_word2vec() {
    Data.wid = {};
    Data.ws.map((e, i) => Data.wid[e] = i);
    Data.tensor = {};
    Data.tensor.vs = tf.tensor(Data.vs);

    word2vec.w2v = function(w) {
        return Data.vs[Data.wid[w]];
    }

    word2vec.w2ws = function(w, topn=5) {
        return tf.tidy(() => {
            let v = word2vec.w2v(w);
            v = tf.tensor(v);
            return word2vec.v2ws(v, topn);
        });
    }

    word2vec.v2ws = function(v, topn=5) {
        return tf.tidy(() => {
            v = v.reshape([v.shape[0], 1]);
            let p = tf.matMul(Data.tensor.vs, v);
            p = p.flatten();
            let ids = tf.topk(p, topn).indices;
            ids = ids.arraySync();
            p = p.arraySync();
            // remove the first
            ids.shift();
            let ws = ids.map(i => Data.ws[i]);
            let ps = ids.map(i => p[i]);
            return {ws:ws, ps:ps};
        });
    }
}

function between(vs, gs) {
    return tf.tidy(() => {
        vs = tf.tensor(vs);
        gs = tf.tensor(gs);
        vs = vs.transpose();
        vs = tf.mul(vs, gs);
        v = vs.sum(-1);
        return tf.div(v, tf.norm(v));
    });
}
