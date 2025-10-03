import { useState, useEffect } from 'react';
import type { IndicatorDefinition, IndicatorConfig } from '../types';
import { indicatorApi } from '../services/api';

interface IndicatorSelectorProps {
  selectedIndicators: IndicatorConfig[];
  onChange: (indicators: IndicatorConfig[]) => void;
}

export default function IndicatorSelector({ selectedIndicators, onChange }: IndicatorSelectorProps) {
  const [availableIndicators, setAvailableIndicators] = useState<IndicatorDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchIndicators = async () => {
      try {
        const indicators = await indicatorApi.list();
        setAvailableIndicators(indicators);
      } catch (err) {
        setError('Failed to load indicators');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchIndicators();
  }, []);

  const addIndicator = (indicatorId: string) => {
    const indicator = availableIndicators.find(i => i.id === indicatorId);
    if (!indicator) return;

    // Create default params object
    const defaultParams: Record<string, any> = {};
    indicator.params.forEach(param => {
      defaultParams[param.name] = param.default;
    });

    const newIndicator: IndicatorConfig = {
      id: indicatorId,
      params: defaultParams,
    };

    onChange([...selectedIndicators, newIndicator]);
  };

  const removeIndicator = (index: number) => {
    const updated = selectedIndicators.filter((_, i) => i !== index);
    onChange(updated);
  };

  const updateIndicatorParam = (index: number, paramName: string, value: any) => {
    const updated = [...selectedIndicators];
    updated[index] = {
      ...updated[index],
      params: {
        ...updated[index].params,
        [paramName]: value,
      },
    };
    onChange(updated);
  };

  const getIndicatorDefinition = (id: string): IndicatorDefinition | undefined => {
    return availableIndicators.find(i => i.id === id);
  };

  if (loading) {
    return <div className="text-gray-500">Loading indicators...</div>;
  }

  if (error) {
    return <div className="text-red-600">{error}</div>;
  }

  // Group indicators by category
  const groupedIndicators = availableIndicators.reduce((acc, indicator) => {
    if (!acc[indicator.category]) {
      acc[indicator.category] = [];
    }
    acc[indicator.category].push(indicator);
    return acc;
  }, {} as Record<string, IndicatorDefinition[]>);

  return (
    <div className="space-y-4">
      {/* Add Indicator Dropdown */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Add Indicator
        </label>
        <select
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          onChange={(e) => {
            if (e.target.value) {
              addIndicator(e.target.value);
              e.target.value = '';
            }
          }}
          defaultValue=""
        >
          <option value="">Select an indicator...</option>
          {Object.entries(groupedIndicators).map(([category, indicators]) => (
            <optgroup key={category} label={category.charAt(0).toUpperCase() + category.slice(1)}>
              {indicators.map(indicator => (
                <option key={indicator.id} value={indicator.id}>
                  {indicator.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>

      {/* Selected Indicators */}
      {selectedIndicators.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-700">Selected Indicators</h3>
          {selectedIndicators.map((selectedIndicator, index) => {
            const definition = getIndicatorDefinition(selectedIndicator.id);
            if (!definition) return null;

            return (
              <div key={index} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <div className="flex justify-between items-start mb-3">
                  <h4 className="font-medium text-gray-900">{definition.name}</h4>
                  <button
                    type="button"
                    onClick={() => removeIndicator(index)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Remove
                  </button>
                </div>

                {/* Indicator Parameters */}
                {definition.params.length > 0 && (
                  <div className="grid grid-cols-2 gap-3">
                    {definition.params.map(param => (
                      <div key={param.name}>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          {param.name.charAt(0).toUpperCase() + param.name.slice(1)}
                        </label>
                        <input
                          type="number"
                          value={selectedIndicator.params[param.name] ?? param.default}
                          onChange={(e) => {
                            const value = param.type === 'float'
                              ? parseFloat(e.target.value)
                              : parseInt(e.target.value);
                            updateIndicatorParam(index, param.name, value);
                          }}
                          min={param.min}
                          max={param.max}
                          step={param.type === 'float' ? '0.1' : '1'}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                        />
                        {(param.min !== undefined || param.max !== undefined) && (
                          <p className="text-xs text-gray-500 mt-1">
                            {param.min !== undefined && param.max !== undefined
                              ? `${param.min}-${param.max}`
                              : param.min !== undefined
                              ? `min: ${param.min}`
                              : `max: ${param.max}`}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {selectedIndicators.length === 0 && (
        <div className="text-sm text-gray-500 text-center py-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          No indicators selected. Add indicators from the dropdown above.
        </div>
      )}
    </div>
  );
}
