import datetime

from pypoanal import backtester, portfolio_calculators

# @pytest.mark.skip
def test_backtest_MCAP():
    start_date = datetime.date(2009,5,10)
    backtest_end_date = datetime.date(2022, 1, 15)
    rebalance_period = datetime.timedelta(days=360)
    rebalance_dates = backtester.compute_rebalance_dates(start_date, backtest_end_date, rebalance_period)
    initial_cash = 10 ** 4
    tickers_list = ['ABCL','ABGI','ACIW','ACNB','ADPT','AEAC','AEL','AERC','AFIB','AGI','AKAM','AKIC','AL','AM','AMH',
     'AMNB','ANIX','ANSS','APLS','ARCK','ARKR','ARTL','ASND','AT','ATHA','ATIF','ATLC','ATXS','AUPH','BAC','BAX',
     'BBW','BCX','BHLB','BITF','BKN','BKU','BLBX','BLDE','BNTX','BON','BRKL','BSMX','BURL','BXRX','C','CAJ','CASI',
     'CASY','CB','CBAY','CBIO','CBTX','CCEP','CCL','CCM','CEN','CEO','CFIV','CGC','CHCI','CIM','CLAQ','CLFD',
     'CLS','CMCO','COKE','COLM','COO','CPSH','CPSS','CSGS','CSX','CVLT','CYAD','CYAN','CZZ','DAIO','DAWN',
     'DBGI','DCI','DEI','DFH','DHR','DMRC','DNAD','DNB','DNOW','DPRO','DUNE','EBTC','ECR','EMBK',
     'EMES','ENPH','EOD','ESRT','ESS','ESV','ETH','FBIZ','FCAP','FDS','FE','FEMY','FEO','FFIC','FMBI','FOUN',
     'FOX','FRO','FRSG','FRSX','FSTX','FUND','GBL','GCO','GDRX','GET','GGG','GLBL','GPN','GPRO','GPS','GRNQ',
     'GSMG','GTLB','GVA','HBM','HCCI','HCJ','HGSH','HHS','HITI','HIW','HLT','HNP','HQH','HQL','HUIZ','HXL',
     'HYI','IGIC','IIF','INDB','INFI','IOBT','IRCP','ISSC','IVZ','JAGX','JAMF','JDD','JTD','KPTI','KRKR','KTOS','LBC',
     'LIFE','LILA','LION','LIXT','LNSR','LTRN','LTRPB','LW','MA','MAA','MAIN','MCLD','MDB','MKTW','MMMB',
     'MNTS','MNTV','MPC','MQY','MTB','MYPS','NBHC','NCBS','NDSN','NEXI','NFLX','NI','NTP','NWE','NXGL','NYMTN','OIIM',
     'OPNT','OSBC','OUT','PAYC','PCG','PFM','PKE','PKX','PLAY','PLMR','PNF','PRSR','PSO','PSX','PT','PTE','PTGX',
     'PXLW','RBBN','RE','RELL','RGT','RKLB','RNER','RNW','RPD','SAL','SAP','SATS','SC','SCMA','SCPL','SCU','SEER',
     'SFET','SGHT','SHBI','SIDU','SILK','SJM','SLAB','SNPX','SOPH','SPRO','SPWH','SRLP','SSBK','SSSS','STIM',
     'STLD','SU','SVRA','SYBT','TASK','TDC','TKC','TOL','TOWN','TPTX','TRVI','TS','TSI','TURN','TXG','UIS',
     'UPS','USAP','VC','VEON','VFF','VIPS','VPG','VRA','VTN','WABC','WASH','WATT','WEN','WFC','WHLR','WILC',
     'WMS','WORX','WSM','WY','XELA','XRX','XTLB','YELL','YMAB','ZEAL','ZG','ZM','ZYXI']
    calculators = {'MCAP': portfolio_calculators.compute_mcap_weights}
    values_df, _, _ = backtester.compare_calculators_for_periodic_rebalance(calculators, tickers_list,
                                                                            initial_cash, rebalance_dates, progress_bar=True)
    assert len(values_df) > 0


# @pytest.mark.skip
def test_backtest_ECOV():
    start_date = datetime.date(2009,5,10)
    backtest_end_date = datetime.date(2022, 1, 15)
    rebalance_period = datetime.timedelta(days=360)
    # rebalance_dates = pd.date_range(start=start_date,end=backtest_end_date, freq=pd.tseries.offsets.DateOffset(months=12))
    rebalance_dates = backtester.compute_rebalance_dates(start_date,backtest_end_date,rebalance_period)
    initial_cash = 10 ** 4
    tickers_list = ['ABCL','ABGI','ACIW','ACNB','ADPT','AEAC','AEL','AERC','AFIB','AGI','AKAM','AKIC','AL','AM','AMH',
     'AMNB','ANIX','ANSS','APLS','ARCK','ARKR','ARTL','ASND','AT','ATHA','ATIF','ATLC','ATXS','AUPH','BAC','BAX',
     'BBW','BCX','BHLB','BITF','BKN','BKU','BLBX','BLDE','BNTX','BON','BRKL','BSMX','BURL','BXRX','C','CAJ','CASI',
     'CASY','CB','CBAY','CBIO','CBTX','CCEP','CCL','CCM','CEN','CEO','CFIV','CGC','CHCI','CIM','CLAQ','CLFD',
     'CLS','CMCO','COKE','COLM','COO','CPSH','CPSS','CSGS','CSX','CVLT','CYAD','CYAN','CZZ','DAIO','DAWN',
     'DBGI','DCI','DEI','DFH','DHR','DMRC','DNAD','DNB','DNOW','DPRO','DUNE','EBTC','ECR','EMBK',
     'EMES','ENPH','EOD','ESRT','ESS','ESV','ETH','FBIZ','FCAP','FDS','FE','FEMY','FEO','FFIC','FMBI','FOUN',
     'FOX','FRO','FRSG','FRSX','FSTX','FUND','GBL','GCO','GDRX','GET','GGG','GLBL','GPN','GPRO','GPS','GRNQ',
     'GSMG','GTLB','GVA','HBM','HCCI','HCJ','HGSH','HHS','HITI','HIW','HLT','HNP','HQH','HQL','HUIZ','HXL',
     'HYI','IGIC','IIF','INDB','INFI','IOBT','IRCP','ISSC','IVZ','JAGX','JAMF','JDD','JTD','KPTI','KRKR','KTOS','LBC',
     'LIFE','LILA','LION','LIXT','LNSR','LTRN','LTRPB','LW','MA','MAA','MAIN','MCLD','MDB','MKTW','MMMB',
     'MNTS','MNTV','MPC','MQY','MTB','MYPS','NBHC','NCBS','NDSN','NEXI','NFLX','NI','NTP','NWE','NXGL','NYMTN','OIIM',
     'OPNT','OSBC','OUT','PAYC','PCG','PFM','PKE','PKX','PLAY','PLMR','PNF','PRSR','PSO','PSX','PT','PTE','PTGX',
     'PXLW','RBBN','RE','RELL','RGT','RKLB','RNER','RNW','RPD','SAL','SAP','SATS','SC','SCMA','SCPL','SCU','SEER',
     'SFET','SGHT','SHBI','SIDU','SILK','SJM','SLAB','SNPX','SOPH','SPRO','SPWH','SRLP','SSBK','SSSS','STIM',
     'STLD','SU','SVRA','SYBT','TASK','TDC','TKC','TOL','TOWN','TPTX','TRVI','TS','TSI','TURN','TXG','UIS',
     'UPS','USAP','VC','VEON','VFF','VIPS','VPG','VRA','VTN','WABC','WASH','WATT','WEN','WFC','WHLR','WILC',
     'WMS','WORX','WSM','WY','XELA','XRX','XTLB','YELL','YMAB','ZEAL','ZG','ZM','ZYXI']
    calculators = {'ECOV': portfolio_calculators.compute_expcov_weights}
    values_df, _, _ = backtester.compare_calculators_for_periodic_rebalance(calculators,
                                                                            tickers_list,
                                                                            initial_cash,
                                                                            rebalance_dates,
                                                                            progress_bar=True)
    assert len(values_df) > 0