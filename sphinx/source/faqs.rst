.. _faqs:

FAQ
===

Below you can find answers to commonly asked questions as well as how to cite FACET.

Commonly asked questions
------------------------

If you don't see your answer below you could also try posting
on `stackoverflow <https://stackoverflow.com/>`_.

1. **What if I find a bug or have an idea for a new feature?**

    For bug reports or feature requests please use our
    `GitHub issue tracker <https://github.com/BCG-Gamma/facet/issues>`_.
    For any other enquiries please feel free to contact us at FacetTeam@bcg.com.

2. **How does FACET's novel algorithm calculate pairwise feature redundancy and synergy?**

    Please keep an eye out for our scientific publication coming soon. In the meantime
    please feel free to explore the
    `GAMMAscope article <https://medium.com/bcggamma/gamma-facet-a-new-approach-for-universal-explanations-of-machine-learning-models-b566877e7812>`__
    to get an introduction to using the algorithm.

3. **How can I contribute?**

    We welcome contributors! If you have minor changes in mind that would like to
    contribute, please feel free to create a pull request and be sure to follow the
    :ref:`developer guidelines<contribution-guide>`.
    For large or extensive changes please feel free to open an
    issue, or reach out to us at FacetTeam@bcg.com to discuss.

4. **How can I perform standard plotting of**
   `SHAP <https://shap.readthedocs.io/en/latest/>`__ **values?**

    You can do this by creating an output of SHAP values from the fit LearnerInspector.

    .. code-block:: Python

        # run inspector
        inspector = LearnerInspector(
            pipeline=clf_selector.best_estimator_,
            n_jobs=-3,
            verbose=False,
        ).fit(sample=sample)

        # get shap values and associated data
        shap_data = inspector.shap_plot_data()
        shap.summary_plot(shap_values=shap_data.shap_values, features=shap_data.features)



5. **How can I extract CV performance from the LearnerSelector to create my
   own summaries or figures?**

    You can extract the desired information as a data frame from the fitted
    LearnerSelector object.

    .. code-block:: Python

        # after fitting a selector
        cv_result_df = selector.summary_report()


6. **Can I use a custom scoring function with the LearnerSelector?**

    The LearnerSelector works in a similar fashion to *scikit-learn*'s
    `gridsearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
    so much of the functionality is equivalent. You can pass a custom scoring
    function much as you would for gridsearchCV.

    .. code-block:: Python

        # define your own custom scorer, in this case Huber loss with delta=3
        import numpy as np
        from sklearn.metrics import make_scorer

        def huber_loss(y_true, y_pred, delta=3):
            diff = y_true - y_pred
            abs_diff = np.abs(diff)
            loss = np.where(abs_diff < delta, (diff**2)/2, delta*abs_diff - (delta**2)/2)
            return np.sum(loss)

        my_score = make_scorer(huber_loss, greater_is_better=False)

        # use the LearnerSelector with custom scorer and get summary report
        selector = LearnerSelector(
            searcher_type=GridSearchCV,
            parameter_space=ps,
            cv=cv_iterator,
            scoring=my_score
        ).fit(
            sample=FACET_sample_object
        )
        selector.summary_report()

    You can see more information on custom scoring with *scikit-learn*
    `here <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`__.


7. **How can I generate standard** *scikit-learn* **summaries for classifiers, such as a
   classification report, confusion matrix or ROC curve?**

    You can extract the fitted best scored model from the LearnerSelector and
    then generate these summaries as you normally would in your *scikit-learn*
    workflow.

    .. code-block:: Python

        # get your ranking object
        selector = LearnerSelector(
            searcher_type=GridSearchCV,
            parameter_space=ps,
            cv=cv_iterator,
            scoring="accuracy"
        ).fit(
            sample=FACET_sample
        )

        # obtain required quantities
        y_pred = selector.best_estimator_.predict(FACET_sample.features)
        y_prob = selector.best_estimator_.predict_proba(FACET_sample.features)[1]
        y_true = FACET_sample.target

        # generate outputs of interest
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )

        # classification report
        print(classification_report(y_true, y_pred))

        # confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cf_matrix).plot()

        # roc curve
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
        auc_val = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
        ax.plot(fpr, tpr, color='lime', label=r'AUC = %0.2f' % (auc_val), lw=2, alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc='lower right')

Citation
--------
If you use FACET in your work we would appreciate if you cite the package.

Bibtex entry::

     @manual{
     title={FACET},
     author={FACET Team at BCG GAMMA},
     year={2021},
     note={Python package version 1.1.0}
     }
