from exp import Exp

if __name__ == "__main__":
    A = "age"
    inject = None
    inject = "synthetic"
    # inject = {A: -0.2}
    exp = Exp("synthetic2", "GroupClassBalance", inject)
    m_train, m_test, m_baseline = exp.one_exp()
    print("")
    print("Train:")
    print("Accuracy: %f" % m_train.accuracy())
    print("EOD: %f" % m_train.eod())
    print("AOD: %f" % m_train.aod())
    # print("Cross EOD: %f" % m_train.cross_eod(A))
    # print("Cross FOD: %f" % m_train.cross_fod(A))
    # print("Cross AOD: %f" % m_train.cross_aod(A))
    print("")
    print("Test:")
    print("Accuracy: %f" % m_test.accuracy())
    print("EOD: %f" % m_test.eod())
    print("AOD: %f" % m_test.aod())
    # print("Cross EOD: %f" % m_test.cross_eod(A))
    # print("Cross FOD: %f" % m_test.cross_fod(A))
    # print("Cross AOD: %f" % m_test.cross_aod(A))
    # print("")
    # print("Baseline:")
    # print("Accuracy: %f" % m_baseline.accuracy())
    # print("EOD: %f" % m_baseline.eod(A))
    # print("AOD: %f" % m_baseline.aod(A))
    # print("Cross EOD: %f" % m_baseline.cross_eod(A))
    # print("Cross FOD: %f" % m_baseline.cross_fod(A))
    # print("Cross AOD: %f" % m_baseline.cross_aod(A))
    # print("")
    # print("Test - Baseline:")
    # # tpr_test = m_test.tprs()
    # # tpr_baseline = m_baseline.tprs()
    # # eod = {key: tpr_test[key] - tpr_baseline[key] for key in tpr_baseline}
    # # aos_test = m_test.aos()
    # # aos_baseline = m_baseline.aos()
    # # aod = {key: aos_test[key] - aos_baseline[key] for key in aos_baseline}
    # print("EOD: %f" % (m_test.eod(A) - m_baseline.eod(A)))
    # print("AOD: %f" % (m_test.aod(A) - m_baseline.aod(A)))
