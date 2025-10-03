import { useState, useEffect } from 'react';
import { indicatorApi } from '../services/api';

interface CodeEditorProps {
  code: string;
  onChange: (code: string) => void;
}

export default function CodeEditor({ code, onChange }: CodeEditorProps) {
  const [templates, setTemplates] = useState<Record<string, string>>({});
  const [blankTemplate, setBlankTemplate] = useState<string>('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);

  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const [templatesData, blankData] = await Promise.all([
          indicatorApi.getTemplates(),
          indicatorApi.getBlankTemplate(),
        ]);
        setTemplates(templatesData);
        setBlankTemplate(blankData.template);

        // Set blank template as default if code is empty
        if (!code) {
          onChange(blankData.template);
        }
      } catch (err) {
        console.error('Failed to load templates:', err);
      }
    };

    fetchTemplates();
  }, []);

  const validateCode = async () => {
    if (!code.trim()) {
      setValidationError('Code cannot be empty');
      return;
    }

    setValidating(true);
    setValidationError(null);

    try {
      const result = await indicatorApi.validateCode(code);
      if (result.valid) {
        setValidationError(null);
        alert('✓ Code validation successful!');
      } else {
        setValidationError(result.error || 'Validation failed');
      }
    } catch (err: any) {
      setValidationError(err.response?.data?.error || 'Validation failed');
    } finally {
      setValidating(false);
    }
  };

  const loadTemplate = (templateName: string) => {
    if (templateName === 'blank') {
      onChange(blankTemplate);
    } else if (templates[templateName]) {
      onChange(templates[templateName]);
    }
    setShowTemplates(false);
  };

  return (
    <div className="space-y-4">
      {/* Template Selector */}
      <div className="flex justify-between items-center">
        <label className="block text-sm font-medium text-gray-700">
          Custom Python Strategy Code
        </label>
        <button
          type="button"
          onClick={() => setShowTemplates(!showTemplates)}
          className="text-sm text-blue-600 hover:text-blue-800"
        >
          {showTemplates ? 'Hide Templates' : 'Load Template'}
        </button>
      </div>

      {showTemplates && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Select a Template</h4>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => loadTemplate('blank')}
              className="px-3 py-2 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50 text-left"
            >
              <div className="font-medium">Blank Template</div>
              <div className="text-xs text-gray-500">Start from scratch</div>
            </button>
            {Object.keys(templates).map(templateName => (
              <button
                key={templateName}
                type="button"
                onClick={() => loadTemplate(templateName)}
                className="px-3 py-2 text-sm bg-white border border-gray-300 rounded hover:bg-gray-50 text-left"
              >
                <div className="font-medium capitalize">{templateName.replace('_', ' ')}</div>
                <div className="text-xs text-gray-500">Pre-built strategy</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Code Editor */}
      <div>
        <textarea
          value={code}
          onChange={(e) => onChange(e.target.value)}
          className="w-full h-96 px-3 py-2 font-mono text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Write your custom strategy code here..."
          spellCheck={false}
        />
        <div className="flex justify-between items-start mt-2">
          <div className="text-xs text-gray-500 space-y-1">
            <p>• Use <code className="bg-gray-100 px-1 rounded">price_data</code> DataFrame with OHLCV columns</p>
            <p>• Return a Series with 'BUY', 'SELL', or 'HOLD' signals</p>
            <p>• Access indicators via <code className="bg-gray-100 px-1 rounded">indicator_data</code> dict</p>
            <p>• Available: pandas (pd), numpy (np), indicators service</p>
          </div>
          <button
            type="button"
            onClick={validateCode}
            disabled={validating}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {validating ? 'Validating...' : 'Validate Code'}
          </button>
        </div>
      </div>

      {/* Validation Error */}
      {validationError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-start">
            <span className="text-red-600 font-medium text-sm mr-2">Error:</span>
            <pre className="text-sm text-red-700 whitespace-pre-wrap font-mono">{validationError}</pre>
          </div>
        </div>
      )}

      {/* Code Guidelines */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Code Guidelines</h4>
        <ul className="text-xs text-gray-700 space-y-1">
          <li>• Function must be named <code className="bg-gray-100 px-1 rounded">generate_signals</code></li>
          <li>• Accept parameters: <code className="bg-gray-100 px-1 rounded">price_data, indicator_data</code></li>
          <li>• Return a pandas Series with same length as price_data</li>
          <li>• Each value must be 'BUY', 'SELL', or 'HOLD'</li>
          <li>• No file I/O, no imports, no system calls allowed</li>
          <li>• Code runs in a sandboxed environment for security</li>
        </ul>
      </div>

      {/* Example */}
      <details className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <summary className="text-sm font-medium text-gray-900 cursor-pointer">
          Show Example
        </summary>
        <pre className="mt-2 text-xs font-mono bg-white p-3 rounded border border-gray-300 overflow-x-auto">
{`def generate_signals(price_data, indicator_data):
    """
    Simple RSI strategy example
    """
    signals = pd.Series('HOLD', index=price_data.index)

    # Get RSI from indicator_data
    if 'RSI' in indicator_data:
        rsi = indicator_data['RSI']

        # Buy when RSI < 30 (oversold)
        signals[rsi < 30] = 'BUY'

        # Sell when RSI > 70 (overbought)
        signals[rsi > 70] = 'SELL'

    return signals`}
        </pre>
      </details>
    </div>
  );
}
