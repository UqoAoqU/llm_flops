import unittest

from benchmark_engine.correctness import Tolerance, TopKComparator, normalize_output


class Array:
    device="cpu"; layout="strided"
    def __init__(self, values, dtype): self.values=values; self.dtype=dtype; self.shape=(len(values),)
    def stride(self): return (1,)
    def reshape(self,*_): return self
    def tolist(self): return self.values

def result(indices, scores): return normalize_output({"indices":Array(indices,"int64"), "scores":Array(scores,"float32")})


class TopKComparatorTests(unittest.TestCase):
    def test_configuration_is_validated_at_construction(self):
        for universe in (0,-1,True):
            with self.assertRaises(ValueError): TopKComparator(universe_size=universe)
        for name,value in (("indices_path",""),("scores_path","")):
            with self.assertRaises(ValueError): TopKComparator(**{name:value})
        with self.assertRaises(TypeError): TopKComparator(score_tolerance="loose")
        with self.assertRaises(ValueError): TopKComparator(score_tolerance=Tolerance(-1,0))

    def test_ordered_unordered_metrics(self):
        ref=result([1,2,3],[.9,.8,.7]); cand=result([2,1,4],[.9,.8,.7])
        compared=TopKComparator(ordered=False,universe_size=5).compare(ref,cand)
        self.assertFalse(compared.passed); self.assertAlmostEqual(compared.metrics["precision_at_k"],2/3)
        self.assertAlmostEqual(compared.metrics["jaccard"],.5)
        reordered=result([2,1,3],[.8,.9,.7])
        self.assertTrue(TopKComparator(ordered=False).compare(ref,reordered).passed)

    def test_tie_aware_order_is_legal(self):
        ref=result([1,2,3],[1.,1.,.5]); cand=result([2,1,3],[1.,1.,.5])
        self.assertFalse(TopKComparator().compare(ref,cand).passed)
        self.assertTrue(TopKComparator(tie_aware=True).compare(ref,cand).passed)

    def test_duplicate_and_out_of_bounds_are_diagnosed(self):
        compared=TopKComparator(ordered=False,universe_size=4).compare(result([1,2],[1.,.5]),result([4,4],[1.,.5]))
        self.assertFalse(compared.passed)
        self.assertIn("candidate contains duplicate index",compared.metrics["invalid"])
        self.assertIn("index out of bounds",compared.metrics["invalid"])

    def test_fraction_bool_and_negative_indices_are_invalid(self):
        for indices, reason in (([1.5,2],"not an exact integer"),([True,2],"not an exact integer"),([-1,2],"negative")):
            compared=TopKComparator(ordered=False,universe_size=None).compare(result([1,2],[.9,.8]),result(indices,[.9,.8]))
            self.assertFalse(compared.passed)
            self.assertTrue(any(reason in item for item in compared.metrics["invalid"]))

    def test_scores_must_match_indices_and_follow_index_semantics(self):
        ref=result([1,2],[.9,.8])
        self.assertTrue(TopKComparator(ordered=False).compare(ref,result([2,1],[.8,.9])).passed)
        self.assertFalse(TopKComparator(ordered=False).compare(ref,result([2,1],[.9,.8])).passed)
        short=normalize_output({"indices":Array([1,2],"int64"),"scores":Array([.9],"float32")})
        compared=TopKComparator(ordered=False).compare(short,short)
        self.assertFalse(compared.passed); self.assertIn("score length does not match indices",compared.metrics["invalid"])
        self.assertFalse(TopKComparator(tie_aware=True).compare(short,short).passed)


if __name__ == "__main__": unittest.main()
