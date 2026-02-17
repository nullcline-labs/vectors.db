//! Metadata filtering engine for search queries.
//!
//! Evaluates [`FilterClause`](crate::api::models::FilterClause) predicates against document metadata.
//! Supports `must` (AND) and `must_not` (AND-NOT) conditions with
//! operators: `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in`.

use crate::document::MetadataValue;
use crate::filter_types::{FilterClause, FilterCondition, FilterOperator};
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(pairs: Vec<(&str, MetadataValue)>) -> HashMap<String, MetadataValue> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn cond(field: &str, op: FilterOperator, value: serde_json::Value) -> FilterCondition {
        FilterCondition {
            field: field.to_string(),
            op,
            value: Some(value),
            values: None,
        }
    }

    fn in_cond(field: &str, values: Vec<serde_json::Value>) -> FilterCondition {
        FilterCondition {
            field: field.to_string(),
            op: FilterOperator::In,
            value: None,
            values: Some(values),
        }
    }

    #[test]
    fn test_eq_string() {
        let metadata = meta(vec![("color", MetadataValue::String("red".into()))]);
        let filter = FilterClause {
            must: vec![cond("color", FilterOperator::Eq, json!("red"))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_eq_string_mismatch() {
        let metadata = meta(vec![("color", MetadataValue::String("blue".into()))]);
        let filter = FilterClause {
            must: vec![cond("color", FilterOperator::Eq, json!("red"))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_ne_operator() {
        let metadata = meta(vec![("status", MetadataValue::String("active".into()))]);
        let filter = FilterClause {
            must: vec![cond("status", FilterOperator::Ne, json!("deleted"))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_gt_integer() {
        let metadata = meta(vec![("age", MetadataValue::Integer(25))]);
        let filter = FilterClause {
            must: vec![cond("age", FilterOperator::Gt, json!(18))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_lt_float() {
        let metadata = meta(vec![("score", MetadataValue::Float(0.5))]);
        let filter = FilterClause {
            must: vec![cond("score", FilterOperator::Lt, json!(0.9))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_gte_boundary() {
        let metadata = meta(vec![("count", MetadataValue::Integer(10))]);
        let filter = FilterClause {
            must: vec![cond("count", FilterOperator::Gte, json!(10))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_lte_boundary() {
        let metadata = meta(vec![("count", MetadataValue::Integer(10))]);
        let filter = FilterClause {
            must: vec![cond("count", FilterOperator::Lte, json!(10))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_in_operator() {
        let metadata = meta(vec![("lang", MetadataValue::String("it".into()))]);
        let filter = FilterClause {
            must: vec![in_cond("lang", vec![json!("en"), json!("it"), json!("fr")])],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_in_operator_miss() {
        let metadata = meta(vec![("lang", MetadataValue::String("de".into()))]);
        let filter = FilterClause {
            must: vec![in_cond("lang", vec![json!("en"), json!("it")])],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_must_not() {
        let metadata = meta(vec![("status", MetadataValue::String("deleted".into()))]);
        let filter = FilterClause {
            must: vec![],
            must_not: vec![cond("status", FilterOperator::Eq, json!("deleted"))],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_must_and_must_not_combined() {
        let metadata = meta(vec![
            ("color", MetadataValue::String("red".into())),
            ("size", MetadataValue::Integer(5)),
        ]);
        let filter = FilterClause {
            must: vec![cond("color", FilterOperator::Eq, json!("red"))],
            must_not: vec![cond("size", FilterOperator::Gt, json!(10))],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_missing_field_returns_false() {
        let metadata = meta(vec![]);
        let filter = FilterClause {
            must: vec![cond("missing", FilterOperator::Eq, json!("anything"))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_boolean_eq() {
        let metadata = meta(vec![("active", MetadataValue::Boolean(true))]);
        let filter = FilterClause {
            must: vec![cond("active", FilterOperator::Eq, json!(true))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_empty_filter_matches_all() {
        let metadata = meta(vec![("any", MetadataValue::String("value".into()))]);
        let filter = FilterClause {
            must: vec![],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    // ── None value branches (all operators) ──────────────────────────

    fn none_cond(field: &str, op: FilterOperator) -> FilterCondition {
        FilterCondition {
            field: field.to_string(),
            op,
            value: None,
            values: None,
        }
    }

    #[test]
    fn test_eq_none_value_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(1))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::Eq)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_ne_none_value_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(1))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::Ne)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_gt_none_value_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(1))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::Gt)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_lt_none_value_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(1))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::Lt)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_gte_none_value_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(1))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::Gte)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_lte_none_value_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(1))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::Lte)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_in_none_values_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::String("a".into()))]);
        let filter = FilterClause {
            must: vec![none_cond("x", FilterOperator::In)],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    // ── Type mismatch in json_eq ─────────────────────────────────────

    #[test]
    fn test_eq_string_vs_number_mismatch() {
        let metadata = meta(vec![("x", MetadataValue::String("hello".into()))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Eq, json!(42))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_eq_bool_vs_string_mismatch() {
        let metadata = meta(vec![("x", MetadataValue::Boolean(true))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Eq, json!("true"))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_eq_integer_vs_string_mismatch() {
        let metadata = meta(vec![("x", MetadataValue::Integer(42))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Eq, json!("42"))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_eq_float_vs_string_mismatch() {
        let metadata = meta(vec![("x", MetadataValue::Float(3.14))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Eq, json!("3.14"))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    // ── Non-numeric comparison (json_cmp returns None) ───────────────

    #[test]
    fn test_gt_with_string_metadata_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::String("hello".into()))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Gt, json!(10))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_lt_with_bool_metadata_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Boolean(true))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Lt, json!(10))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }

    // ── Float equality ───────────────────────────────────────────────

    #[test]
    fn test_eq_float_metadata_with_number() {
        let metadata = meta(vec![("score", MetadataValue::Float(0.5))]);
        let filter = FilterClause {
            must: vec![cond("score", FilterOperator::Eq, json!(0.5))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    #[test]
    fn test_eq_integer_with_float_json() {
        // Integer(10) == json!(10.0) via f64 comparison
        let metadata = meta(vec![("x", MetadataValue::Integer(10))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Eq, json!(10.0))],
            must_not: vec![],
        };
        assert!(matches_filter(&metadata, &filter));
    }

    // ── Comparison with non-numeric JSON value ───────────────────────

    #[test]
    fn test_gt_integer_vs_string_json_returns_false() {
        let metadata = meta(vec![("x", MetadataValue::Integer(10))]);
        let filter = FilterClause {
            must: vec![cond("x", FilterOperator::Gt, json!("5"))],
            must_not: vec![],
        };
        assert!(!matches_filter(&metadata, &filter));
    }
}
