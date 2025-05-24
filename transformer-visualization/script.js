document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    const tabs = {
        'overview-btn': 'overview',
        'encoder-btn': 'encoder',
        'decoder-btn': 'decoder',
        'attention-btn': 'attention',
        'multihead-btn': 'multihead',
        'demo-btn': 'demo'
    };

    Object.keys(tabs).forEach(btnId => {
        document.getElementById(btnId).addEventListener('click', function() {
            // Update button states
            Object.keys(tabs).forEach(id => {
                document.getElementById(id).classList.remove('active');
            });
            this.classList.add('active');

            // Update visualization states
            Object.values(tabs).forEach(id => {
                document.getElementById(id).classList.remove('active');
            });
            document.getElementById(tabs[btnId]).classList.add('active');

            // Initialize specific visualizations
            if (btnId === 'attention-btn') {
                initializeAttentionVisualization();
            } else if (btnId === 'demo-btn') {
                initializeTranslationDemo();
            }
        });
    });

    // Initialize attention weights matrix
    function initializeAttentionVisualization() {
        const weightMatrix = document.getElementById('attention-weights-matrix');
        weightMatrix.innerHTML = '';

        // Sample attention weights - higher values for semantically related words
        const attentionWeights = [
            [0.6, 0.1, 0.2, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.3],
            [0.1, 0.1, 0.3, 0.5]
        ];

        const tokens = ['我', '爱', '机器', '学习'];
        
        // Create header row
        const headerRow = document.createElement('div');
        headerRow.className = 'weight-header';
        headerRow.style.display = 'grid';
        headerRow.style.gridTemplateColumns = 'repeat(' + (tokens.length + 1) + ', 1fr)';
        
        const cornerCell = document.createElement('div');
        cornerCell.style.textAlign = 'center';
        cornerCell.style.padding = '5px';
        headerRow.appendChild(cornerCell);
        
        tokens.forEach(token => {
            const headerCell = document.createElement('div');
            headerCell.textContent = token;
            headerCell.style.textAlign = 'center';
            headerCell.style.padding = '5px';
            headerCell.style.fontWeight = 'bold';
            headerRow.appendChild(headerCell);
        });
        
        weightMatrix.appendChild(headerRow);

        // Create matrix with row headers
        for (let i = 0; i < tokens.length; i++) {
            const row = document.createElement('div');
            row.style.display = 'grid';
            row.style.gridTemplateColumns = 'repeat(' + (tokens.length + 1) + ', 1fr)';
            
            // Add row header
            const rowHeader = document.createElement('div');
            rowHeader.textContent = tokens[i];
            rowHeader.style.textAlign = 'center';
            rowHeader.style.padding = '5px';
            rowHeader.style.fontWeight = 'bold';
            row.appendChild(rowHeader);
            
            for (let j = 0; j < tokens.length; j++) {
                const cell = document.createElement('div');
                cell.className = 'weight-cell';
                cell.textContent = attentionWeights[i][j].toFixed(1);
                
                // Color intensity based on weight value
                const opacity = 0.2 + attentionWeights[i][j] * 0.8;
                cell.style.backgroundColor = `rgba(52, 152, 219, ${opacity})`;
                cell.style.color = opacity > 0.6 ? 'white' : 'black';
                
                // Add hover effect to show connections
                cell.addEventListener('mouseenter', function() {
                    const fromToken = document.querySelectorAll('.matrix-content .token')[i];
                    const toToken = document.querySelectorAll('.matrix-content .token')[j];
                    
                    fromToken.classList.add('highlight');
                    toToken.classList.add('highlight');
                    this.classList.add('pulse');
                });
                
                cell.addEventListener('mouseleave', function() {
                    document.querySelectorAll('.matrix-content .token').forEach(token => {
                        token.classList.remove('highlight');
                    });
                    this.classList.remove('pulse');
                });
                
                row.appendChild(cell);
            }
            
            weightMatrix.appendChild(row);
        }
    }

    // Initialize translation demo
    function initializeTranslationDemo() {
        const demoMatrix = document.getElementById('demo-attention-matrix');
        demoMatrix.innerHTML = '';
        
        const chineseTokens = ['动物', '没有', '过', '马路', '，', '因为', '它', '太', '累', '了'];
        const englishTokens = ['The', 'animal', 'did', 'not', 'cross', 'the', 'road', 'because', 'it', 'was', 'too', 'tired'];
        
        // Sample attention weights for "它" (focused on index 6)
        // Higher weights for "动物" (index 0) showing the attention mechanism
        // understanding that "它" refers to "动物"
        const attentionWeights = [
            0.4,  // 动物 - high attention
            0.05, // 没有
            0.05, // 过
            0.1,  // 马路
            0.05, // ，
            0.15, // 因为
            0.1,  // 它 (self-attention)
            0.03, // 太
            0.05, // 累
            0.02  // 了
        ];
        
        // Create grid cells for attention visualization
        for (let i = 0; i < chineseTokens.length; i++) {
            const cell = document.createElement('div');
            cell.className = 'attention-cell';
            cell.dataset.index = i;
            cell.dataset.token = chineseTokens[i];
            cell.dataset.weight = attentionWeights[i];
            
            // Set cell opacity based on attention weight
            cell.style.opacity = 0.2 + attentionWeights[i] * 0.8;
            
            demoMatrix.appendChild(cell);
        }
        
        // Play demo button
        document.getElementById('play-demo').addEventListener('click', function() {
            playTranslationDemo();
        });
        
        // Highlight "它" connections button
        document.getElementById('highlight-it').addEventListener('click', function() {
            highlightItConnections();
        });
    }

    // Play translation demo animation
    function playTranslationDemo() {
        const inputTokens = document.querySelectorAll('.demo-input span');
        const outputTokens = document.querySelectorAll('.demo-output span');
        const attentionCells = document.querySelectorAll('.attention-cell');
        
        // Reset any previous highlights
        inputTokens.forEach(token => token.classList.remove('highlighted'));
        outputTokens.forEach(token => token.classList.remove('highlighted'));
        
        // Sequence animation
        let delay = 0;
        const step = 300;
        
        // Animate input tokens
        inputTokens.forEach((token, index) => {
            setTimeout(() => {
                inputTokens.forEach(t => t.classList.remove('highlighted'));
                token.classList.add('highlighted');
            }, delay);
            delay += step;
        });
        
        // Pause between input and output
        delay += 500;
        
        // Animate output tokens
        outputTokens.forEach((token, index) => {
            setTimeout(() => {
                outputTokens.forEach(t => t.classList.remove('highlighted'));
                token.classList.add('highlighted');
            }, delay);
            delay += step;
        });
        
        // End animation - reset highlights
        setTimeout(() => {
            inputTokens.forEach(token => token.classList.remove('highlighted'));
            outputTokens.forEach(token => token.classList.remove('highlighted'));
        }, delay + 500);
    }

    // Highlight "它" connections
    function highlightItConnections() {
        const inputTokens = document.querySelectorAll('.demo-input span');
        const outputTokens = document.querySelectorAll('.demo-output span');
        const attentionCells = document.querySelectorAll('.attention-cell');
        
        // Reset any previous highlights
        inputTokens.forEach(token => token.classList.remove('highlighted'));
        outputTokens.forEach(token => token.classList.remove('highlighted'));
        
        // Highlight "它" in input
        inputTokens[6].classList.add('highlighted'); // "它" is at index 6
        
        // Highlight "it" in output
        outputTokens[8].classList.add('highlighted'); // "it" is at index 8
        
        // Highlight "动物" in input (showing connection)
        inputTokens[0].classList.add('highlighted'); // "动物" is at index 0
        
        // Update attention matrix visual
        attentionCells.forEach(cell => {
            const index = parseInt(cell.dataset.index);
            
            // Emphasize attention for "动物" and "它"
            if (index === 0 || index === 6) {
                cell.style.opacity = 1;
                cell.classList.add('pulse');
            } else {
                cell.style.opacity = 0.2;
                cell.classList.remove('pulse');
            }
        });
        
        // Update current focus text
        document.getElementById('current-focus').textContent = '它';
    }

    // Additional interactive elements for encoder/decoder blocks
    document.querySelectorAll('.block').forEach(block => {
        block.addEventListener('click', function() {
            const isEncoder = this.parentElement.className === 'encoder-blocks';
            const tabButton = isEncoder ? 'encoder-btn' : 'decoder-btn';
            
            // Switch to corresponding tab
            document.getElementById(tabButton).click();
            
            // Highlight this block
            this.classList.add('highlight');
            setTimeout(() => {
                this.classList.remove('highlight');
            }, 1000);
        });
    });

    // Initialize with overview tab active
    document.getElementById('overview-btn').click();
});
