//! Metadata filter types for search queries.
//!
//! Defines the filter clause structure used by storage and search modules
//! for pre-filtering and post-filtering during vector and hybrid search.

use serde::Deserialize;

/// Metadata filter clause with `must` (AND) and `must_not` (AND-NOT) conditions.
#[derive(Debug, Deserialize, Clone)]
pub struct FilterClause {
    #[serde(default)]
    pub must: Vec<FilterCondition>,
    #[serde(default)]
    pub must_not: Vec<FilterCondition>,
}

/// A single filter condition on a metadata field.
#[derive(Debug, Deserialize, Clone)]
pub struct FilterCondition {
    pub field: String,
    pub op: FilterOperator,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub values: Option<Vec<serde_json::Value>>,
}

/// Comparison operator for filter conditions.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum FilterOperator {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    In,
}
