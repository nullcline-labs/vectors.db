//! Metadata filtering engine for search queries.
//!
//! Evaluates [`FilterClause`](crate::api::models::FilterClause) predicates against document metadata.
//! Supports `must` (AND) and `must_not` (AND-NOT) conditions with
//! operators: `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in`.

use crate::api::models::{FilterClause, FilterCondition, FilterOperator};
use crate::document::MetadataValue;
use std::collections::HashMap;

/// Check if a document's metadata matches the given filter clause.
/// `must` conditions are AND-ed; `must_not` conditions are AND-NOT-ed.
pub fn matches_filter(metadata: &HashMap<String, MetadataValue>, filter: &FilterClause) -> bool {
    for cond in &filter.must {
        if !evaluate_condition(metadata, cond) {
            return false;
        }
    }
    for cond in &filter.must_not {
        if evaluate_condition(metadata, cond) {
            return false;
        }
    }
    true
}

fn evaluate_condition(metadata: &HashMap<String, MetadataValue>, cond: &FilterCondition) -> bool {
    let field_value = match metadata.get(&cond.field) {
        Some(v) => v,
        None => return false,
    };

    match cond.op {
        FilterOperator::Eq => {
            if let Some(ref val) = cond.value {
                json_eq(field_value, val)
            } else {
                false
            }
        }
        FilterOperator::Ne => {
            if let Some(ref val) = cond.value {
                !json_eq(field_value, val)
            } else {
                false
            }
        }
        FilterOperator::Gt => {
            if let Some(ref val) = cond.value {
                json_cmp(field_value, val).is_some_and(|o| o == std::cmp::Ordering::Greater)
            } else {
                false
            }
        }
        FilterOperator::Lt => {
            if let Some(ref val) = cond.value {
                json_cmp(field_value, val).is_some_and(|o| o == std::cmp::Ordering::Less)
            } else {
                false
            }
        }
        FilterOperator::Gte => {
            if let Some(ref val) = cond.value {
                json_cmp(field_value, val).is_some_and(|o| o != std::cmp::Ordering::Less)
            } else {
                false
            }
        }
        FilterOperator::Lte => {
            if let Some(ref val) = cond.value {
                json_cmp(field_value, val).is_some_and(|o| o != std::cmp::Ordering::Greater)
            } else {
                false
            }
        }
        FilterOperator::In => {
            if let Some(ref vals) = cond.values {
                vals.iter().any(|v| json_eq(field_value, v))
            } else {
                false
            }
        }
    }
}

/// Compare a MetadataValue with a serde_json::Value for equality.
fn json_eq(meta: &MetadataValue, json: &serde_json::Value) -> bool {
    match (meta, json) {
        (MetadataValue::String(s), serde_json::Value::String(js)) => s == js,
        (MetadataValue::Boolean(b), serde_json::Value::Bool(jb)) => b == jb,
        (MetadataValue::Integer(i), serde_json::Value::Number(n)) => {
            n.as_i64().is_some_and(|ni| *i == ni)
                || n.as_f64()
                    .is_some_and(|nf| (*i as f64 - nf).abs() < f64::EPSILON)
        }
        (MetadataValue::Float(f), serde_json::Value::Number(n)) => {
            n.as_f64().is_some_and(|nf| (*f - nf).abs() < f64::EPSILON)
        }
        _ => false,
    }
}

/// Compare a MetadataValue with a serde_json::Value for ordering.
fn json_cmp(meta: &MetadataValue, json: &serde_json::Value) -> Option<std::cmp::Ordering> {
    let meta_f = match meta {
        MetadataValue::Integer(i) => *i as f64,
        MetadataValue::Float(f) => *f,
        _ => return None,
    };
    let json_f = json.as_f64()?;
    meta_f.partial_cmp(&json_f)
}
